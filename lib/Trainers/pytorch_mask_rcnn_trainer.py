# Python Modules
import numpy as np
import datetime
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
from lib.utils.general import prepare_input, prepare_input_mask_rcnn
from lib.utils.logger import log
from lib.utils.evaluation_metrics import roc_auc_plot  # (y_true, y_pred)
from lib.utils.evaluation_metrics import ct_mask_pred_image_review, box_whisker_list_input
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
            # print(epoch, self.args.n_epochs)

            for param_group in self.optimizer.param_groups:
                lr_show = param_group['lr']
            log.info('\n########################################################################')
            log.info(f"Training epoch: {epoch}, Learning rate: {lr_show:.8f}")

            self.train_epoch_alex(epoch)

            if self.do_validation:
                log.info(f"Validation epoch: {epoch}")
                log.info(len(self.valid_data_loader))  # something is wrong here!
                self.validate_epoch_alex(epoch)
                # print('did it go through?')

            # Set up for visual evaluation
            if epoch % 10 == 0:
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
                        # Gathering input data; prepare_input sends to gpu
                        preprocess_input = {k: v[0] for k, v in input_tuple[1].items()}
                        input_tuple[1] = preprocess_input
                        input_tensor, target = prepare_input_mask_rcnn(input_tuple=input_tuple, args=self.args)

                        #images = list(image.to(device) for image in images)
                        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        # adjusting because I think the code above is for mulitple batch size
                        targets = [{k: v.to(torch.device('cuda')) for k, v in target.items()}]

                        self.model.eval()
                        pred = self.model(input_tensor)
                        pred = {k: v.cpu() for k, v in pred[0].items()}
                        pred = pred['masks']

                        pred_thres = torch.where(pred > self.args.pred_threshold, 1, 0)

                        # pred looks to have separate instances. We need to add them all together
                        pred = pred_thres.cpu().numpy()
                        if len(pred) > 0:
                            pred_store = np.zeros((np.shape(pred)[1],
                                                   np.shape(pred)[2],
                                                   np.shape(pred)[3]))
                            for p in range(len(pred)):
                                pred_store += pred[p, ...]
                                # pred_store += np.where(pred[p, ...] > self.args.pred_threshold, 1, 0)
                            del pred
                            pred = pred_store
                            # pred = torch.Tensor(pred)

                        pred = np.where(pred > self.args.pred_threshold, 1, 0)

                        # will need to do the same for the target as we separated the mask into instances
                        # storing variable has different indices on purpose compared to those used for prediction (targets_store)
                        targets_compiled = targets[0]['masks'].cpu().numpy()
                        if len(targets_compiled) > 0 and len(np.shape(targets_compiled)) == 4:
                            targets_store = np.zeros((np.shape(targets_compiled)[0],
                                                      np.shape(targets_compiled)[2],
                                                      np.shape(targets_compiled)[3]))
                            for t in range(len(targets_compiled)):
                                targets_store += targets_compiled[0, t, ...]
                            del targets_compiled
                            targets_compiled = targets_store

                        if np.random.randint(0, 100) > 96 and len(pred) > 0:

                            if '25d' in self.args.short_note or 'mask_rcnn' in self.args.short_note:
                                input_t = input_tensor.cpu().numpy()[0, 0, ...].reshape(256, 256)

                            else:
                                input_t = input_tensor.cpu().numpy()[0, ...].reshape(256, 256)

                            # 0 because batch size is 1 otherwise have to rethink approach
                            ct_mask_pred_image_review(input_t,
                                                      targets_compiled[0, ...],
                                                      pred[0, ...].reshape(256, 256),
                                                      save_fig_name=f"val_mask_rcnn_pred_{batch_idx}.png",
                                                      view=False,
                                                      fig_storage_dir=os.path.join(save_model_epoch_dir, str(epoch)))

                        # print('pred: ', np.shape(pred))
                        # print('targets_compiled: ', np.shape(targets_compiled))

                        if len(pred) > 0:
                            dice_metric = dice_metric_(torch.Tensor(pred[0, ...].reshape(256, 256)),
                                                       torch.Tensor(targets_compiled[0, ...]))
                            interim_val_pred.append(tuple((targets_compiled[0, ...].max(), dice_metric.cpu().numpy())))

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
                # Gathering input data; prepare_input sends to gpu
                preprocess_input = {k: v[0] for k, v in input_tuple[1].items()}
                input_tuple[1] = preprocess_input
                input_tensor, target = prepare_input_mask_rcnn(input_tuple=input_tuple, args=self.args)

                # adjusting because I think the code above is for mulitple batch size
                targets = [{k: v.to(torch.device('cuda')) for k, v in target.items()}]

                self.model.eval()
                pred = self.model(input_tensor)
                pred = {k: v.cpu() for k, v in pred[0].items()}
                pred = pred['masks']

                pred_thres = torch.where(pred > self.args.pred_threshold, 1, 0)

                # pred looks to have separate instances. We need to add them all together
                pred = pred_thres.cpu().numpy()
                if len(pred) > 0:
                    pred_store = np.zeros((np.shape(pred)[1],
                                           np.shape(pred)[2],
                                           np.shape(pred)[3]))
                    for p in range(len(pred)):
                        pred_store += pred[p, ...]
                        # pred_store += np.where(pred[p, ...] > self.args.pred_threshold, 1, 0)
                    del pred
                    pred = pred_store
                    # pred = torch.Tensor(pred)
                pred = np.where(pred > self.args.pred_threshold, 1, 0)

                # will need to do the same for the target as we separated the mask into instances
                # storing variable has different indices on purpose compared to those used for prediction (targets_store)
                targets_compiled = targets[0]['masks'].cpu().numpy()
                if len(targets_compiled) > 0 and len(np.shape(targets_compiled)) == 4:
                    targets_store = np.zeros((np.shape(targets_compiled)[0],
                                              np.shape(targets_compiled)[2],
                                              np.shape(targets_compiled)[3]))
                    for t in range(len(targets_compiled)):
                        targets_store += targets_compiled[0, t, ...]
                    del targets_compiled
                    targets_compiled = targets_store

                if len(pred) > 0:
                    dice_metric = dice_metric_(torch.Tensor(pred[0, ...].reshape(256, 256)),
                                               torch.Tensor(targets_compiled[0, ...]))
                    final_val_pred.append(tuple((targets_compiled[0, ...].max(), dice_metric.cpu().numpy())))

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
        self.tb_logger.add_figure(f"Box Plot - Validation Dice (Threshold: {self.args.pred_threshold})",
                                  figure=box_plot_fig)
        self.tb_logger.flush()

        # test
        final_test_pred = []

        for batch_idx, input_tuple in enumerate(self.final_test_data_loader):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                preprocess_input = {k: v[0] for k, v in input_tuple[1].items()}
                input_tuple[1] = preprocess_input
                input_tensor, target = prepare_input_mask_rcnn(input_tuple=input_tuple, args=self.args)

                # adjusting because I think the code above is for mulitple batch size
                targets = [{k: v.to(torch.device('cuda')) for k, v in target.items()}]

                self.model.eval()
                pred = self.model(input_tensor)
                pred = {k: v.cpu() for k, v in pred[0].items()}
                pred = pred['masks']

                pred_thres = torch.where(pred > self.args.pred_threshold, 1, 0)

                # pred looks to have separate instances. We need to add them all together
                pred = pred_thres.cpu().numpy()
                if len(pred) > 0:
                    pred_store = np.zeros((np.shape(pred)[1],
                                           np.shape(pred)[2],
                                           np.shape(pred)[3]))
                    for p in range(len(pred)):
                        pred_store += pred[p, ...]
                        # pred_store += np.where(pred[p, ...] > self.args.pred_threshold, 1, 0)
                    del pred
                    pred = pred_store
                    # pred = torch.Tensor(pred)

                pred = np.where(pred > self.args.pred_threshold, 1, 0)

                # will need to do the same for the target as we separated the mask into instances
                # storing variable has different indices on purpose compared to those used for prediction (targets_store)
                targets_compiled = targets[0]['masks'].cpu().numpy()
                if len(targets_compiled) > 0 and len(np.shape(targets_compiled)) == 4:
                    targets_store = np.zeros((np.shape(targets_compiled)[0],
                                              np.shape(targets_compiled)[2],
                                              np.shape(targets_compiled)[3]))
                    for t in range(len(targets_compiled)):
                        targets_store += targets_compiled[0, t, ...]
                    del targets_compiled
                    targets_compiled = targets_store

                if len(pred) > 0:
                    dice_metric = dice_metric_(torch.Tensor(pred[0, ...].reshape(256, 256)),
                                               torch.Tensor(targets_compiled[0, ...]))
                    final_test_pred.append(tuple((targets_compiled[0, ...].max(), dice_metric.cpu().numpy())))

        # splitting dice_metric into should have mask and should not have mask

        dice_mask = [x[1] for x in final_test_pred if x[0] == 1]
        dice_no_mask = [x[1] for x in final_test_pred if x[0] == 0]
        dice_all = [x[1] for x in final_test_pred]

        box_plot_fig = box_whisker_list_input(data=[dice_mask, dice_no_mask, dice_all],
                                              labels=['Mask', 'No Mask', 'All'],
                                              fig_title='Final test Performance')
        self.tb_logger.add_figure(f"Box Plot - Test Dice (Threshold: {self.args.pred_threshold})", figure=box_plot_fig)
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

        log.info('-------------------------------------------------------------------------------------------')

        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            # Gathering input data; prepare_input sends to gpu
            preprocess_input = {k: v[0] for k, v in input_tuple[1].items()}
            input_tuple[1] = preprocess_input
            input_tensor, target = prepare_input_mask_rcnn(input_tuple=input_tuple, args=self.args)

            # adjusting because I think the code above is for mulitple batch size
            targets = [{k: v.to(torch.device('cuda')) for k, v in target.items()}]

            loss_dict = self.model(input_tensor, targets)
            losses = sum(loss for loss in loss_dict.values())

            # need to calculate gradient
            self.model.zero_grad()

            losses.backward()

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
                loss_cum.append(losses.item())
                # loss_thres_cum.append(loss_thres.item())

                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    log.info(f"\tTrain batch {batch_idx + 1} of {len(self.train_data_loader)}")
                    log.info(
                        f"\tLoss: {losses.item()}")
                    log.info(loss_dict)
                    log.info('\t**************************************************************************')
                else:
                    pass

            self.tb_logger.add_scalar(f"training_loss", losses.item(), self.train_count)
            # self.tb_logger.add_scalar(f"training_loss_thres", loss_thres.item(), self.train_count)
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

        # Storing epoch values obtained from batch calculations
        loss_cum = []

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
                # Gathering input data; prepare_input sends to gpu
                preprocess_input = {k: v[0] for k, v in input_tuple[1].items()}
                input_tuple[1] = preprocess_input
                input_tensor, target = prepare_input_mask_rcnn(input_tuple=input_tuple, args=self.args)

                # adjusted from github
                targets = [{k: v.to(torch.device('cuda')) for k, v in target.items()}]

                loss_dict = self.model(input_tensor, targets)
                # print(loss_dict)

                try:
                    losses = sum(loss for loss in loss_dict.values())
                except:
                    pass

                # have to activate model.eval() now or else cant get loss above
                self.model.eval()
                pred = self.model(input_tensor)
                pred = {k: v.cpu() for k, v in pred[0].items()}
                pred = pred['masks']

                pred_thres = torch.where(pred > self.args.pred_threshold, 1, 0)

                # print(targets[0]['masks'].size())
                # print(pred.size())

                # pred looks to have separate instances. We need to add them all together
                pred = pred.cpu().numpy()
                if len(pred) > 0:
                    pred_store = np.zeros((np.shape(pred)[1],
                                           np.shape(pred)[2],
                                           np.shape(pred)[3]))
                    for p in range(len(pred)):
                        pred_store += pred[p, ...]
                        # pred_store += np.where(pred[p, ...] > self.args.pred_threshold, 1, 0)
                    del pred
                    pred = pred_store
                pred = np.where(pred > self.args.pred_threshold, 1, 0)
                    # pred = torch.Tensor(pred)

                # will need to do the same for the target as we separated the mask into instances
                # storing variable has different indices on purpose compared to those used for prediction (targets_store)
                targets_compiled = targets[0]['masks'].cpu().numpy()
                if len(targets_compiled) > 0 and len(np.shape(targets_compiled)) == 4:
                    targets_store = np.zeros((np.shape(targets_compiled)[0],
                                              np.shape(targets_compiled)[2],
                                              np.shape(targets_compiled)[3]))
                    for t in range(len(targets_compiled)):
                        targets_store += targets_compiled[0, t, ...]
                    del targets_compiled
                    targets_compiled = targets_store
                    # targets_compiled = torch.Tensor(pred)

                # print(input_tensor.size())
                # print(np.shape(pred))
                # print(len(pred))
                # print(np.shape(targets_compiled))

                #check to make sure prediction isnt empty
                if not len(pred) == 0 and epoch >= 10:
                    # pass
                    if np.random.randint(0, 1000) > 980:
                        time_stamp = datetime.datetime.now()
                        ttime = time_stamp.strftime('%Y_%m_%d-%H_%M_%S')

                        if '25d' in self.args.short_note or 'mask_rcnn' in self.args.short_note:
                            input_t = input_tensor.cpu().numpy()[0, 0, ...].reshape(256, 256)

                        else:
                            input_t = input_tensor.cpu().numpy()[0, ...].reshape(256, 256)

                        ct_mask_pred_image_review(input_t,
                                                  targets_compiled[0, ...],
                                                  pred[0, ...].reshape(256, 256),
                                                  save_fig_name=f"val_mrcnn_pred_{self.args.short_note}_excludeblanks_{self.args.exclude_blanks}_"
                                                                f"augpercent_{self.args.aug_percent}_epoch_{self.args.n_epochs}_" + ttime + '.png',
                                                  view=False,
                                                  fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/')

                # storing loss and metrics
                try:
                    loss_cum.append(losses.item())
                except:
                    loss_cum.append(np.nan)
                # loss_thres_cum.append(loss_thres.item())

                self.tb_logger.add_scalar(f"val_loss", losses.item(), self.val_count)
                # self.tb_logger.add_scalar(f"val_loss_thres", loss_thres.item(), self.val_count)
                self.tb_logger.flush()

                self.val_count += 1

        # self.val_loss = sum(loss_cum) / len(loss_cum)

        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)
        total_nan = len([x for x in loss_cum if np.isnan(x)])
        loss_cum = [x for x in loss_cum if not np.isnan(x)]
        log.info(
            f"Summary-----Loss: {np.round(np.sum(loss_cum) / len(loss_cum), 4)}, total nan: {total_nan} "
            f"Epoch: {res} (min:seconds)")
        log.info('-------------------------------------------------------------------------------------------')