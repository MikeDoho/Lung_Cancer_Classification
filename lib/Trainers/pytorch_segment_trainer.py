# Python Modules
import numpy as np
import datetime
import glob
import os
import time
import math
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt

# Torch Modules
import torch
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# Self Modules
from lib.utils.general import prepare_input
from lib.utils.logger import log
from lib.utils.evaluation_metrics import roc_auc_plot  # (y_true, y_pred)
from lib.utils.evaluation_metrics import ct_mask_pred_image_review, box_whisker_list_input, ct_mask_pred_25d_image_review
from lib.Loss.dice import DiceLoss, dice_loss
from lib.Loss.dice import dice_metric as dice_metric_


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion_pre, optimizer,
                 train_data_loader=None, valid_data_loader=None, test_data_loader=None,
                 final_valid_data_loader=None, final_test_data_loader=None,
                 lr_scheduler=None, tb_logger=None, criterion_pre2=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion_pre = criterion_pre
        self.criterion_pre2 = criterion_pre2
        self.train_data_loader = train_data_loader
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.final_valid_data_loader = final_valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader
        self.final_test_data_loader = final_test_data_loader
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.save_frequency = 200
        self.start_epoch = 1
        self.val_loss = 0

        self.print_batch_spacing = 50
        self.save_interval = args.save_intervals
        self.tb_logger = tb_logger
        self.train_count = 0
        self.val_count = 0

    def training(self):

        for epoch in range(self.start_epoch, (self.args.n_epochs + 1)):

            for param_group in self.optimizer.param_groups:
                lr_show = param_group['lr']
            log.info('\n########################################################################')
            log.info(f"Training epoch: {epoch}, Learning rate: {lr_show:.8f}")

            self.train_epoch_alex(epoch)

            if self.do_validation:
                log.info(f"Validation epoch: {epoch}")
                self.validate_epoch_alex(epoch)

            # Set up for visual evaluation
            if epoch % 25 == 0:
                save_img_dir = r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/'
                epoch_dir = f"{self.args.short_note}_excludeblanks_{self.args.exclude_blanks}_" \
                            f"augpercent_{self.args.aug_percent}_epoch_{self.args.n_epochs}_" \
                            f"loss_{self.args.loss_select}_filternum_{self.args.unet_feature_num}_thres_" \
                            f"{self.args.pred_threshold}_{self.args.time_stamp}"

                save_model_epoch_dir = os.path.join(save_img_dir, epoch_dir)

                if not os.path.exists(save_model_epoch_dir):
                    os.makedirs(save_model_epoch_dir)
                else:
                    pass

                if not os.path.exists(os.path.join(save_model_epoch_dir, str(epoch))):
                    os.makedirs(os.path.join(save_model_epoch_dir, str(epoch)))
                else:
                    pass

                # validation
                interim_val_pred = []

                for batch_idx, input_tuple in enumerate(self.final_valid_data_loader):
                    with torch.no_grad():
                        input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                        pred = self.model(input_tensor)
                        pred_thresh = torch.where(pred > self.args.pred_threshold, 1, 0)

                        if np.random.randint(0, 100) > 96:

                            if '25d' in self.args.short_note:
                                input_t = input_tensor.cpu().numpy()[0, 0, ...].reshape(256, 256)
                            else:
                                input_t = input_tensor.cpu().numpy()[0, ...].reshape(256, 256)

                            # 0 because batch size is 1 otherwise have to rethink approach
                            ct_mask_pred_image_review(input_t,
                                                      target.cpu().numpy()[0, ...].reshape(256, 256),
                                                      pred_thresh.cpu().numpy()[0, ...].reshape(256, 256),
                                                      save_fig_name=f"val_pred_{batch_idx}.png",
                                                      view=False,
                                                      fig_storage_dir=os.path.join(save_model_epoch_dir, str(epoch)))

                        dice_metric = dice_metric_(pred_thresh, target)
                        interim_val_pred.append(tuple((target.cpu().numpy().max(), dice_metric.cpu().numpy())))

                # splitting dice_metric into should have mask and should not have mask
                dice_mask = [x[1] for x in interim_val_pred if x[0] == 1]
                dice_no_mask = [x[1] for x in interim_val_pred if x[0] == 0]
                dice_all = [x[1] for x in interim_val_pred]

                box_plot_fig = box_whisker_list_input(data=[dice_mask, dice_no_mask, dice_all],
                                                      labels=['Mask', 'No Mask', 'All'],
                                                      fig_title=f"Interim Validation Performance-{epoch}")
                box_plot_fig.savefig(os.path.join(os.path.join(save_model_epoch_dir, str(epoch)),
                                                  f"box_whisker_{epoch}_threshold_{self.args.pred_threshold}.png"))
                plt.close(box_plot_fig)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # # comment out for speed test
            # if epoch % self.save_frequency == 0:
            #     self.model.save_checkpoint(self.args.save,
            #                            epoch, self.val_loss,
            #                            optimizer=self.optimizer)

        # EVALUATION AFTER TRAINING HAS BEEN COMPLETED
        # validation
        final_val_pred = []

        for batch_idx, input_tuple in enumerate(self.final_valid_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                pred = self.model(input_tensor)
                pred_thresh = torch.where(pred > self.args.pred_threshold, 1, 0)
                # pred_thres = pred_thresh(pred)
                dice_metric = dice_metric_(pred_thresh, target)
                final_val_pred.append(tuple((target.cpu().numpy().max(), dice_metric.cpu().numpy())))

        # splitting dice_metric into should have mask and should not have mask
        # print(final_val_pred)
        dice_mask = [x[1] for x in final_val_pred if x[0] == 1]
        dice_no_mask = [x[1] for x in final_val_pred if x[0] == 0]
        dice_all = [x[1] for x in final_val_pred]

        # print([dice_mask, dice_no_mask, dice_all])
        # print(np.concatenate[np.array(dice_mask), np.array(dice_no_mask), np.array(dice_all)])

        box_plot_fig = box_whisker_list_input(data=[dice_mask, dice_no_mask, dice_all],
                                              labels=['Mask', 'No Mask', 'All'],
                                              fig_title='Final Validation Performance')
        self.tb_logger.add_figure(f"Box Plot Validation Dice Threshold {self.args.pred_threshold}",
                                  figure=box_plot_fig)
        self.tb_logger.flush()

        # test
        final_test_pred = []

        for batch_idx, input_tuple in enumerate(self.final_test_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                pred = self.model(input_tensor)
                pred_thresh = torch.where(pred > self.args.pred_threshold, 1, 0)
                dice_metric = dice_metric_(pred_thresh, target)
                final_test_pred.append(tuple((target.cpu().numpy().max(), dice_metric.cpu().numpy())))

        # splitting dice_metric into should have mask and should not have mask

        dice_mask = [x[1] for x in final_test_pred if x[0] == 1]
        dice_no_mask = [x[1] for x in final_test_pred if x[0] == 0]
        dice_all = [x[1] for x in final_test_pred]

        box_plot_fig = box_whisker_list_input(data=[dice_mask, dice_no_mask, dice_all],
                                              labels=['Mask', 'No Mask', 'All'],
                                              fig_title='Final test Performance')
        self.tb_logger.add_figure(f"Box Plot Test Dice Threshold {self.args.pred_threshold}", figure=box_plot_fig)
        self.tb_logger.flush()
        # if '25d' not in self.args.short_note:
        #     self.tb_logger.close()

        # Creating testing environment where if 2.5D then will use first prediction as first input
        if '25d' in self.args.short_note:

            # going to make a HUGE ASSUMPTION that I am loading the slices properly AND that there are only 90 slices
            # per patient!!!!
            print('HUGE ASSUMPTIONS SHOULD CHECK IF TRYING TO IMPLEMENT OTHER CASES!! .....lazy')
            print('NOT DONE YET')
            print('seems like the assumption holds')

            # have to reload data again. will use same approach from data loader.
            train_image = glob.glob(os.path.join(self.args.train_dataset_path, '*.npy'))

            image = train_image
            print('len of all image directories: ', len(image))
            image = [x for x in image if x.split('/')[-1].split('_')[0] not in self.args.exclude_mrns]

            print('len of all image directories after excluding mrns: ', len(image))
            labels = glob.glob(os.path.join(self.args.train_label_path, '*.npy'))
            labels = [x for x in labels if x.split('/')[-1].split('_')[0] not in self.args.exclude_mrns]

            image_val = [x for x in image if tuple((x.split('/')[-1].split('_')[0], x.split('/')[-1].split('_')[-2])) in
                         self.args.val_zip_list and not x.split('/')[-1].split('.')[0].split('_')[-1].isnumeric()]
            labels_val = [x for x in labels if
                          tuple((x.split('/')[-1].split('_')[0], x.split('/')[-1].split('_')[-2])) in
                          self.args.val_zip_list]

            # this is sorting by patient and by slice; specification 0, 1, 2, 3, 4, ...., 88, 89
            image_val = sorted(image_val,
                               key=lambda x: (int(x.split('/')[-1].split('_')[0]),
                                              int(x.split('/')[-1].split('_')[-2])))
            labels_val = sorted(labels_val,
                                key=lambda x: (int(x.split('/')[-1].split('_')[0]),
                                               int(x.split('/')[-1].split('_')[-2])))

            # creating list w/ tuples (full dir for image, full dir for label)
            new_load_val = list(zip(image_val, labels_val))

            # validation
            final_val_pred_true = []

            for i, dir_tuple in enumerate(new_load_val):
                with torch.no_grad():

                    # if i == 0:
                    #     # creating dumb checking function to make sure loading slices of the same mrns
                    #     # need to add this for the initial patient otherwise this statement will be in the if/else
                    #     # statement below
                    #     check_mrn.append(dir_tuple[1].split('/')[-1].split('_')[0])

                    if dir_tuple[1].split('/')[-1].split('_')[-2] == '0':

                        # checking to make sure mrns are only from 1 patient
                        mrn_to_check = dir_tuple[1].split('/')[-1].split('_')[0]

                        # will need to load initial image; if initial slice then set prior label to zero matrix
                        img_load = np.load(dir_tuple[0])
                        label_load = np.load(dir_tuple[1])

                        # need to normalize the data!!
                        # Simple normalization
                        img_load = np.clip(img_load, -1000, 2000)
                        min_ = np.array(np.min(img_load), dtype=float)
                        max_ = np.array(np.max(img_load), dtype=float)
                        img_load = (img_load - min_) / (max_ - min_ + 1e-12)

                        # creating zero prior matrix if slice == 0
                        prior_label = np.zeros_like(label_load)
                        new_img_load = np.zeros((2, np.shape(img_load)[0], np.shape(img_load)[1]))

                        # combined img+label to get new input
                        new_img_load[0, ...] = img_load
                        new_img_load[1, ...] = prior_label

                        # input_tuple = tuple(new_img_load, label_load)

                    else:
                        # appending mrn to check later
                        # assert str(check_mrn) == str(dir_tuple[1].split('/')[-1].split('_')[0]), \
                        #     f"need to make sure slices are loading appropriately\n" \
                        #     f"loaded: {dir_tuple[1].split('/')[-1].split('_')[0]} when should load " \
                        #     f"{mrn_to_check}"

                        img_load = np.load(dir_tuple[0])
                        label_load = np.load(dir_tuple[1])
                        # store_prior_label = np.zeros_like(label_load)

                        # need to normalize the data!!
                        # Simple normalization
                        img_load = np.clip(img_load, -1000, 2000)
                        min_ = np.array(np.min(img_load), dtype=float)
                        max_ = np.array(np.max(img_load), dtype=float)
                        img_load = (img_load - min_) / (max_ - min_ + 1e-12)

                        # creating zero prior matrix if slice == 0
                        # prior_label = np.zeros_like(label_load)
                        new_img_load = np.zeros((2, np.shape(img_load)[0], np.shape(img_load)[1]))

                        # combined img+label to get new input
                        new_img_load[0, ...] = img_load
                        new_img_load[1, ...] = store_prior_label

                    # print('check', dir_tuple[1].split('/')[-1].split('_')[0],
                    #       dir_tuple[1].split('/')[-1].split('_')[-2])

                    input_tuple = tuple((torch.Tensor(np.expand_dims(new_img_load, axis=0)),
                                         torch.Tensor(np.expand_dims(label_load, axis=0))))

                    input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                    pred = self.model(input_tensor)
                    pred_thresh = torch.where(pred > self.args.pred_threshold, 1, 0)
                    store_prior_label = pred_thresh.cpu().numpy().reshape(256, 256)
                    # print('val pred thresh size: ', pred_thresh.size())
                    # print('val target size: ', target.size())
                    # pred_thres = pred_thresh(pred)
                    dice_metric = dice_metric_(pred_thresh, target)
                    # print('val dice: ', dice_metric)
                    final_val_pred_true.append(tuple((target.cpu().numpy().max(), dice_metric.cpu().numpy())))

                    # view the predictions
                    if np.random.randint(0, 100) > 80:
                        input_t = input_tensor.cpu().numpy()[0, 0, ...].reshape(256, 256)
                        input_p = input_tensor.cpu().numpy()[0, 1, ...].reshape(256, 256)

                        ct_mask_pred_25d_image_review(input_t,
                                                      target.cpu().numpy()[0, ...].reshape(256, 256),
                                                      pred.cpu().numpy()[0, ...].reshape(256, 256),
                                                      input_p,
                                                      save_fig_name=f"val_pred_{i}_{self.args.short_note}_excludeblanks_{self.args.exclude_blanks}_"
                                                                f"augpercent_{self.args.aug_percent}_epoch_{self.args.n_epochs}.png",
                                                      view=False,
                                                      fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/_25D/validation')

            # splitting dice_metric into should have mask and should not have mask
            # print(final_val_pred)
            dice_mask = [x[1] for x in final_val_pred_true if x[0] == 1]
            dice_no_mask = [x[1] for x in final_val_pred_true if x[0] == 0]
            dice_all = [x[1] for x in final_val_pred_true]

            theList = dice_all
            N = 90
            patient_dice = [theList[n:n + N] for n in range(0, len(theList), N)]

            # print([dice_mask, dice_no_mask, dice_all])
            # print(np.concatenate[np.array(dice_mask), np.array(dice_no_mask), np.array(dice_all)])

            box_plot_fig = box_whisker_list_input(data=[dice_mask, dice_no_mask, dice_all],
                                                  labels=['Mask', 'No Mask', 'All'],
                                                  fig_title=f"True Val, total pt median/mean: "
                                                            f"{np.round(np.median(patient_dice), 3)}/{np.round(np.mean(patient_dice), 3)} "
                                                            f"+/- {np.round(np.std(patient_dice), 3)}")
            self.tb_logger.add_figure(f"True_Val_Dice_Threshold_{self.args.pred_threshold}",
                                      figure=box_plot_fig)
            self.tb_logger.flush()


            ## Repeating process above for test data

            image_test = [x for x in image if tuple((x.split('/')[-1].split('_')[0], x.split('/')[-1].split('_')[-2])) in
                         self.args.test_zip_list and not x.split('/')[-1].split('.')[0].split('_')[-1].isnumeric()]
            labels_test = [x for x in labels if
                          tuple((x.split('/')[-1].split('_')[0], x.split('/')[-1].split('_')[-2])) in
                          self.args.test_zip_list]

            # this is sorting by patient and by slice; specification 0, 1, 2, 3, 4, ...., 88, 89
            image_test = sorted(image_test,
                               key=lambda x: (int(x.split('/')[-1].split('_')[0]),
                                              int(x.split('/')[-1].split('_')[-2])))
            labels_test = sorted(labels_test,
                                key=lambda x: (int(x.split('/')[-1].split('_')[0]),
                                               int(x.split('/')[-1].split('_')[-2])))

            # creating list w/ tuples (full dir for image, full dir for label)
            new_load_test = list(zip(image_test, labels_test))

            # validation
            final_test_pred_true = []

            for i, dir_tuple in enumerate(new_load_test):
                with torch.no_grad():

                    if dir_tuple[1].split('/')[-1].split('_')[-2] == '0':

                        # will need to load initial image; if initial slice then set prior label to zero matrix
                        img_load = np.load(dir_tuple[0])
                        label_load = np.load(dir_tuple[1])

                        # need to normalize the data!!
                        # Simple normalization
                        img_load = np.clip(img_load, -1000, 2000)
                        min_ = np.array(np.min(img_load), dtype=float)
                        max_ = np.array(np.max(img_load), dtype=float)
                        img_load = (img_load - min_) / (max_ - min_ + 1e-12)

                        # creating zero prior matrix if slice == 0
                        prior_label = np.zeros_like(label_load)
                        new_img_load = np.zeros((2, np.shape(img_load)[0], np.shape(img_load)[1]))

                        # combined img+label to get new input
                        new_img_load[0, ...] = img_load
                        new_img_load[1, ...] = prior_label

                        # input_tuple = tuple(new_img_load, label_load)

                    else:
                        # appending mrn to check later
                        # assert str(check_mrn) == str(dir_tuple[1].split('/')[-1].split('_')[0]), \
                        #     f"need to make sure slices are loading appropriately\n" \
                        #     f"loaded: {dir_tuple[1].split('/')[-1].split('_')[0]} when should load " \
                        #     f"{mrn_to_check}"

                        img_load = np.load(dir_tuple[0])
                        label_load = np.load(dir_tuple[1])
                        # store_prior_label = np.zeros_like(label_load)

                        # need to normalize the data!!
                        # Simple normalization
                        img_load = np.clip(img_load, -1000, 2000)
                        min_ = np.array(np.min(img_load), dtype=float)
                        max_ = np.array(np.max(img_load), dtype=float)
                        img_load = (img_load - min_) / (max_ - min_ + 1e-12)

                        # creating zero prior matrix if slice == 0
                        # prior_label = np.zeros_like(label_load)
                        new_img_load = np.zeros((2, np.shape(img_load)[0], np.shape(img_load)[1]))

                        # combined img+label to get new input
                        new_img_load[0, ...] = img_load
                        new_img_load[1, ...] = store_prior_label

                    # print('check', dir_tuple[1].split('/')[-1].split('_')[0],
                    #       dir_tuple[1].split('/')[-1].split('_')[-2])

                    input_tuple = tuple((torch.Tensor(np.expand_dims(new_img_load, axis=0)),
                                         torch.Tensor(np.expand_dims(label_load, axis=0))))

                    input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                    pred = self.model(input_tensor)
                    pred_thresh = torch.where(pred > self.args.pred_threshold, 1, 0)
                    store_prior_label = pred_thresh.cpu().numpy().reshape(256, 256)
                    # pred_thres = pred_thresh(pred)
                    print('test pred thresh size: ', pred_thresh.size())
                    print('test target size: ', target.size())
                    dice_metric = dice_metric_(pred_thresh, target)
                    print('test dice: ', dice_metric)
                    final_test_pred_true.append(tuple((target.cpu().numpy().max(), dice_metric.cpu().numpy())))

                    # view the predictions
                    if np.random.randint(0, 100) > 50:
                        input_t = input_tensor.cpu().numpy()[0, 0, ...].reshape(256, 256)
                        input_p = input_tensor.cpu().numpy()[0, 1, ...].reshape(256, 256)

                        ct_mask_pred_25d_image_review(input_t,
                                                      target.cpu().numpy()[0, ...].reshape(256, 256),
                                                      pred_thresh.cpu().numpy()[0, ...].reshape(256, 256),
                                                      input_p,
                                                      save_fig_name=f"test_pred_{i}_{self.args.short_note}_excludeblanks_{self.args.exclude_blanks}_"
                                                                    f"augpercent_{self.args.aug_percent}_epoch_{self.args.n_epochs}.png",
                                                      view=False,
                                                      fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/_25D/test')

            # splitting dice_metric into should have mask and should not have mask
            # print(final_val_pred)
            dice_mask = [x[1] for x in final_val_pred_true if x[0] == 1]
            dice_no_mask = [x[1] for x in final_val_pred_true if x[0] == 0]
            dice_all = [x[1] for x in final_val_pred_true]

            theList = dice_all
            N = 90
            patient_dice = [theList[n:n + N] for n in range(0, len(theList), N)]
            # print('patient dice')
            # print(np.mean(patient_dice, axis=1))

            # print([dice_mask, dice_no_mask, dice_all])
            # print(np.concatenate[np.array(dice_mask), np.array(dice_no_mask), np.array(dice_all)])

            box_plot_fig = box_whisker_list_input(data=[dice_mask, dice_no_mask, dice_all],
                                                  labels=['Mask', 'No Mask', 'All'],
                                                  fig_title=f"True Test, total pt median/mean: "
                                                            f"{np.round(np.median(patient_dice), 3)}/{np.round(np.mean(patient_dice), 3)} "
                                                            f"+/- {np.round(np.std(patient_dice), 3)}")
            self.tb_logger.add_figure(f"True_Test_Dice_Threshold_{self.args.pred_threshold}",
                                      figure=box_plot_fig)
            self.tb_logger.flush()

        self.tb_logger.close()

    def train_epoch_alex(self, epoch):

        # Creates once at the beginning of training

        def time_report(initial_time, time_name):
            get_time_diff = time.gmtime(time.time() - initial_time)
            readable_time = time.strftime("%M:%S", get_time_diff)
            print(f"{time_name} time: {readable_time} (min:seconds)")
            del get_time_diff
            del readable_time

        epoch_start_time = time.time()
        self.model.train()

        # Storing epoch values obtained from batch calculations
        loss_cum = []
        loss_thres_cum = []

        log.info('-------------------------------------------------------------------------------------------')

        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            # Gathering input data; prepare_input sends to gpu
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

            # may need to turn on if want to train fully but off for transfer learning
            # input_tensor.requires_grad = True

            # Model make prediction
            pred = self.model(input_tensor)

            # Thresholding to evaluate performance
            pred_thresh = torch.nn.Threshold(self.args.pred_threshold, 0)
            # pred_thres = torch.where(pred > self.args.pred_threshold, 1, 0)
            pred_thres = pred_thresh(pred)
            # print(pred_thres.size(), target.size())

            loss_thres = self.criterion_pre(pred_thres, target)

            # calculating loss and metrics
            loss = self.criterion_pre(pred, target)

            if self.criterion_pre2 != None:
                loss2 = self.criterion_pre2(pred, target)
                loss = loss + loss2

            # need to calculate gradient
            self.model.zero_grad()

            loss.backward()

            def clip_gradient(optimizer, grad_clip):
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.clamp_(-grad_clip, grad_clip)

            clip_gradient(self.optimizer, 5)
            self.optimizer.step()

            # Calculating and appending
            with torch.no_grad():
                # storing loss and metrics
                loss_cum.append(loss.item())
                loss_thres_cum.append(loss_thres.item())

                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    log.info(f"\tTrain batch {batch_idx + 1} of {len(self.train_data_loader)}")
                    log.info(
                        f"\tLoss: {loss.item()}")
                    log.info('\t**************************************************************************')
                else:
                    pass

            self.tb_logger.add_scalar(f"training_loss", loss.item(), self.train_count)
            self.tb_logger.add_scalar(f"training_loss_thres", loss_thres.item(), self.train_count)
            self.tb_logger.flush()

            # train count for tensorboard logging
            self.train_count += 1

            if not self.args.ci_test:
                # save model
                if batch_idx == 0 and (epoch * len(self.train_data_loader)) != 0 and (
                        epoch * len(self.train_data_loader)) % self.save_interval == 0:
                    # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = f"{self.args.save_folder}_{self.args.short_note}_epoch_{epoch}_batchidx_{batch_idx}_" \
                                      f"{self.args.single_fold}_of_{self.args.cv_num}_excludeblanks_{self.args.exclude_blanks}_" \
                                      f"augpercent_{self.args.aug_percent}_{self.args.loss_select}_" \
                                      f"threshold_{self.args.pred_threshold}_{self.args.time_stamp}.pth.tar"
                    # model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(self.args.save_folder, epoch, batch_idx)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)

                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_idx))
                    torch.save({
                        'epoch': epoch,
                        'batch_id': batch_idx,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()},
                        model_save_path)

        # Calculating time per epoch
        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)
        log.info(
            f"Summary-----Loss: {np.round(sum(loss_cum) / len(loss_cum), 4)}, Epoch: {res} (min:seconds)")
        log.info('-------------------------------------------------------------------------------------------')

    def validate_epoch_alex(self, epoch):
        self.model.eval()

        # Storing epoch values obtained from batch calculations
        loss_cum = []
        loss_thres_cum = []

        # starting epoch timer
        epoch_start_time = time.time()

        log.info('-------------------------------------------------------------------------------------------')
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            if (batch_idx + 1) % self.print_batch_spacing == 0:
                log.info('*************************************')
                log.info(f"\tValidation batch {batch_idx + 1} of {len(self.valid_data_loader)}")
            else:
                pass

            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                # input_tensor.requires_grad = False

                pred = self.model(input_tensor)
                # pred = (pred > 0.5).float()
                pred_thresh = torch.nn.Threshold(self.args.pred_threshold, 0)
                # pred_thres = torch.where(pred > self.args.pred_threshold, 1, 0)
                pred_thres = pred_thresh(pred)
                loss_thres = self.criterion_pre(pred_thres, target)

                loss = self.criterion_pre(pred, target)

                if self.criterion_pre2 != None:
                    loss2 = self.criterion_pre2(pred, target)
                    loss = loss + loss2

                for i in range(len(target.cpu().numpy())):
                    if target.cpu().numpy()[i, ...].max() == 1 or target.cpu().numpy()[i, ...].max() == 0:
                        # pass
                        if np.random.randint(0, 10000) > 9995:
                            # rand = np.random.randint(0, 100)
                            time_stamp = datetime.datetime.now()
                            ttime = time_stamp.strftime('%Y_%m_%d-%H_%M_%S')
                            # print('shapes')
                            # print('input: ', np.shape(input_tensor.cpu().numpy()))
                            # print('label: ', np.shape(target.cpu().numpy()))
                            # print('pred: ', np.shape(pred.cpu().numpy()))

                            if '25d' in self.args.short_note:
                                input_t = input_tensor.cpu().numpy()[i, 0, ...].reshape(256, 256)
                            else:
                                input_t = input_tensor.cpu().numpy()[i, ...].reshape(256, 256)

                            ct_mask_pred_image_review(input_t,
                                                      target.cpu().numpy()[i, ...].reshape(256, 256),
                                                      pred.cpu().numpy()[i, ...].reshape(256, 256),
                                                      save_fig_name=f"val_pred_{self.args.short_note}_excludeblanks_{self.args.exclude_blanks}_"
                                                                    f"augpercent_{self.args.aug_percent}_epoch_{self.args.n_epochs}_" + ttime + '.png',
                                                      view=False,
                                                      fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/')

                # storing loss and metrics
                loss_cum.append(loss.item())
                loss_thres_cum.append(loss_thres.item())

                self.tb_logger.add_scalar(f"val_loss", loss.item(), self.val_count)
                self.tb_logger.add_scalar(f"val_loss_thres", loss_thres.item(), self.val_count)
                self.tb_logger.flush()

                self.val_count += 1

        self.val_loss = sum(loss_cum) / len(loss_cum)

        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)
        log.info(
            f"Summary-----Loss: {np.round(sum(loss_cum) / len(loss_cum), 4)}, Epoch: {res} (min:seconds)")
        log.info('-------------------------------------------------------------------------------------------')
