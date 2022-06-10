# Python Modules
import numpy as np
import os
import time
import math
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, classification_report

# Torch Modules
import torch
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

# Self Modules
from lib.utils.general import prepare_input
from lib.utils.logger import log
from lib.utils.evaluation_metrics import roc_auc_plot  # (y_true, y_pred)
from lib.Models.clinical_ft_model import clinical_model
from lib.Loading.clinical_data_loader import clinical_data
from lib.Loading.feedtube_pretrain_clinic import FEEDTUBE


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
      E.g. for use with categorical_crossentropy.
      Args:
          y: class vector to be converted into a matrix
              (integers from 0 to num_classes).
          num_classes: total number of classes. If `None`, this would be inferred
            as the (largest number in `y`) + 1.
          dtype: The data type expected by the input. Default: `'float32'`.
      Returns:
          A binary matrix representation of the input. The classes axis is placed
          last.
      Example:
      >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
      >>> a = tf.constant(a, shape=[4, 4])
      >>> print(a)
      tf.Tensor(
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)
      >>> b = tf.constant([.9, .04, .03, .03,
      ...                  .3, .45, .15, .13,
      ...                  .04, .01, .94, .05,
      ...                  .12, .21, .5, .17],
      ...                 shape=[4, 4])
      >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
      >>> print(np.around(loss, 5))
      [0.10536 0.82807 0.1011  1.77196]
      >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
      >>> print(np.around(loss, 5))
      [0. 0. 0. 0.]
      Raises:
          Value Error: If input contains string value
      """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def dr_friendly_measures(outputs, targets):
    with torch.no_grad():
        outputs = torch.argmax(torch.sigmoid(outputs), dim=1)

        try:
            tn, fp, fn, tp = confusion_matrix(targets.cpu().numpy(), outputs.cpu().numpy()).ravel()
            specificity = tn / (tn + fp + 1e-12)
            sensitivity = tp / (tp + fn + 1e-12)
            return specificity, sensitivity
        except:
            return np.nan, np.nan


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dr_friendly_measures_reg(outputs, targets):
    # outputs = np.argmax(sigmoid(outputs))

    assert np.shape(outputs) == np.shape(targets), 'prediction and target outcomes should be same shape'

    try:
        tn, fp, fn, tp = confusion_matrix(targets, outputs).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        return specificity, sensitivity
    except:
        return np.nan, np.nan


def calculate_auc(outputs, targets):
    with torch.no_grad():
        outputs = torch.sigmoid(outputs)
        try:
            auc = roc_auc_score(targets.cpu().numpy(), outputs.type(torch.FloatTensor).cpu().data.numpy()[:, 1])
            return auc
        except:
            return np.nan


def calculate_prauc(outputs, targets):
    with torch.no_grad():
        outputs = torch.sigmoid(outputs)
        try:
            prauc = average_precision_score(targets.cpu().numpy(),
                                            outputs.type(torch.FloatTensor).cpu().data.numpy()[:, 1])
            return prauc
        except:
            return np.nan


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion_pre, optimizer, train_data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, tb_logger=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion_pre = criterion_pre
        self.train_data_loader = train_data_loader
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.save_frequency = 20
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

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # # comment out for speed test
            # if epoch % self.save_frequency == 0:
            #     self.model.save_checkpoint(self.args.save,
            #                            epoch, self.val_loss,
            #                            optimizer=self.optimizer)

            print('\n\n')

        ###WILL NEED TO TRAIN CLINICAL MODEL HERE###
        log.info('loading clinical data')
        x_train_clinic, y_train_clinic = \
            clinical_data(args=self.args,
                          train_mrn_list=self.args.train_mrn_list).load_train_data_batch(
                batch_size=len(self.args.train_mrn_list))
        # log.info('train data: ', np.shape(x_train_clinic))

        x_val_clinic, y_val_clinic = \
            clinical_data(args=self.args,
                          val_mrn_list=self.args.val_mrn_list,
                          train_mrn_list=self.args.train_mrn_list).load_val_data_batch(
                batch_size=len(self.args.val_mrn_list))

        # log.info('val data: ', np.shape(x_val_clinic))

        x_test_clinic, y_test_clinic = \
            clinical_data(args=self.args,
                          train_mrn_list=self.args.train_mrn_list,
                          test_mrn_list=self.args.test_mrn_list).load_test_data_batch(
                batch_size=len(self.args.test_mrn_list))


        clinical_model_ = clinical_model(args=self.args, X=x_train_clinic, y=y_train_clinic,
                                         model_type=self.args.clinical_model_type).create_model()

        y_pred_val_clinic = clinical_model_.predict_proba(x_val_clinic)
        y_pred_train_clinic = clinical_model_.predict_proba(x_train_clinic)
        y_pred_test_clinic = clinical_model_.predict_proba(x_test_clinic)

        log.info('predictions made')

        # Creating list to store prediction information from image and clinic based model
        auc_store = []
        auc_store_clinic = []

        rand_num_cv = np.random.randint(0, 1000)

        ### ASSESSING CLINICAL MODEL ###
        print('\nVALIDATION INFORMATION - CLINICAL BASED MODEL\n')
        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report(np.argmax(y_val_clinic, axis=-1),
                                          np.argmax(y_pred_val_clinic, axis=-1),
                                          target_names=target_names))

        print('\nTRAIN INFORMATION - CLINIC BASED MODEL\n')
        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report(np.argmax(y_train_clinic, axis=-1),
                                          np.argmax(y_pred_train_clinic, axis=-1),
                                          target_names=target_names))

        print('\nTEST INFORMATION - CLINIC BASED MODEL\n')
        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report(np.argmax(y_test_clinic, axis=-1),
                                          np.argmax(y_pred_test_clinic, axis=-1),
                                          target_names=target_names))

        # Clinical model evaluation that will be saved to tensorboard
        # Validation results
        fig_add = roc_auc_plot(y_val_clinic, y_pred_val_clinic, data_title=f"Clinical Model Val AUC")
        self.tb_logger.add_figure(f"Clinical Model Validation AUC", figure=fig_add)
        clinic_spec, clinic_sens = dr_friendly_measures_reg(np.argmax(y_pred_val_clinic, axis=-1), np.argmax(y_val_clinic, axis=-1))
        print('sensitivity: ', clinic_sens, 'specificity: ', clinic_spec)

        self.tb_logger.add_scalar(f"clinic_model_val_spec", clinic_spec, 0)
        self.tb_logger.add_scalar(f"clinic_model_val_sens", clinic_sens, 0)
        self.tb_logger.flush()

        # AUC values gathered for total prediction model
        auc_ = roc_auc_score(y_true=np.argmax(y_val_clinic, axis=-1), y_score=y_pred_val_clinic[:, 1])
        print('clinical model auc_: ', auc_)
        auc_store_clinic.append(auc_)

        # Train results
        fig_add = roc_auc_plot(y_train_clinic, y_pred_train_clinic, data_title=f"Clinical Model Train AUC")
        self.tb_logger.add_figure(f"Clinical Model Train AUC", figure=fig_add)
        clinic_spec, clinic_sens = dr_friendly_measures_reg(np.argmax(y_pred_train_clinic, axis=-1),
                                                            np.argmax(y_train_clinic, axis=-1))

        self.tb_logger.add_scalar(f"clinic_model_train_spec", clinic_spec, 0)
        self.tb_logger.add_scalar(f"clinic_model_train_sens", clinic_sens, 0)
        self.tb_logger.flush()

        # Test results
        fig_add = roc_auc_plot(y_test_clinic, y_pred_test_clinic, data_title=f"Clinical Model Test AUC")
        self.tb_logger.add_figure(f"Clinical Model Test AUC", figure=fig_add)
        clinic_spec, clinic_sens = dr_friendly_measures_reg(np.argmax(y_pred_test_clinic, axis=-1),
                                                            np.argmax(y_test_clinic, axis=-1))

        self.tb_logger.add_scalar(f"clinic_model_test_spec", clinic_spec, 0)
        self.tb_logger.add_scalar(f"clinic_model_test_sens", clinic_sens, 0)
        self.tb_logger.flush()

        ### FINISHED ASSESSING CLINICAL MODEL ###

        ### ASSESSING IMAGE MODEL ###

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        # creating PR curve training/validation
        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                # Model make prediction
                pred = self.model(input_tensor)
                pred_.extend(torch.argmax(torch.sigmoid(pred), dim=1).cpu().numpy())
                log.info(np.array(torch.sigmoid(pred).cpu().numpy()).tolist())
                pred2_.extend(np.array(torch.sigmoid(pred).cpu().numpy()).tolist())

                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

        target2_ = np.array(target2_)
        pred2_ = np.array(pred2_)

        fig_add = roc_auc_plot(target2_, pred2_, data_title='Training ROC')
        self.tb_logger.add_figure(f"Image Model Train AUC", figure=fig_add)

        target_ = torch.from_numpy(np.array(target_).astype('int32'))
        pred_ = torch.from_numpy(np.array(pred_).astype('int32'))

        self.tb_logger.add_pr_curve(f"Image Model Training PR Curve", target_, pred_)
        self.tb_logger.flush()

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                # Model make prediction
                pred = self.model(input_tensor)
                pred_.extend(torch.argmax(torch.sigmoid(pred), dim=1).cpu().numpy())
                pred2_.extend(np.array(torch.sigmoid(pred).cpu().numpy()).tolist())

                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

        target2_ = np.array(target2_)
        pred2_ = np.array(pred2_)

        fig_add = roc_auc_plot(target2_, pred2_, data_title='Validation ROC')
        self.tb_logger.add_figure(f"Image Model Validation AUC", figure=fig_add)

        # AUC values gathered for total prediction model
        auc_ = roc_auc_score(y_true=np.argmax(target2_, axis=-1), y_score=pred2_[:, 1])
        auc_store.append(auc_)

        target_ = torch.from_numpy(np.array(target_).astype('int32'))
        pred_ = torch.from_numpy(np.array(pred_).astype('int32'))

        self.tb_logger.add_pr_curve(f"Image Model Validation PR Curve", target_, pred_)
        self.tb_logger.flush()

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        # Testing
        for batch_idx, input_tuple in enumerate(self.test_data_loader):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                # Model make prediction
                pred = self.model(input_tensor)
                pred_.extend(torch.argmax(torch.sigmoid(pred), dim=1).cpu().numpy())
                pred2_.extend(np.array(torch.sigmoid(pred).cpu().numpy()).tolist())

                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

        target2_ = np.array(target2_)
        pred2_ = np.array(pred2_)

        fig_add = roc_auc_plot(target2_, pred2_, data_title='Test ROC')
        self.tb_logger.add_figure(f"Image Model Test AUC", figure=fig_add)

        target_ = torch.from_numpy(np.array(target_).astype('int32'))
        pred_ = torch.from_numpy(np.array(pred_).astype('int32'))

        self.tb_logger.add_pr_curve(f"Image Model Test PR Curve", target_, pred_)
        self.tb_logger.flush()

        ### FINISHED ASSESSING IMAGE MODEL ###

        ### ASSESSING IMAGE + CLINIC MODEL ###

        # image factor / (image factor + clinic factor)
        alpha_image = auc_store[0] / (auc_store[0] + auc_store_clinic[0])

        # Creating new validation and testing generators for validation
        # VALIDATION
        validation_dataset_eval = FEEDTUBE(self.args, mode='val',
                                           train_path=self.args.train_dataset_path,
                                           val_path=self.args.val_dataset_path,
                                           test_path=self.args.test_dataset_path,
                                           # dataset_path=args.train_dataset_path,
                                           label_path=self.args.val_label_path,
                                           exclude_mrns=self.args.exclude_mrns,
                                           clinic_image_eval=True)
        val_generator_eval = DataLoader(validation_dataset_eval, batch_size=1, shuffle=False, num_workers=0)

        # TESTING
        test_dataset_eval = FEEDTUBE(self.args, mode='test',
                                     train_path=self.args.train_dataset_path,
                                     val_path=self.args.val_dataset_path,
                                     test_path=self.args.test_dataset_path,
                                     # dataset_path=args.train_dataset_path,
                                     label_path=self.args.test_label_path,
                                     exclude_mrns=self.args.exclude_mrns,
                                     clinic_image_eval=True)
        test_generator_eval = DataLoader(test_dataset_eval, batch_size=1, shuffle=False, num_workers=0)

        # Evaluating image and clinic model performance on validation data

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        target2_c = []
        pred2_c = []

        for batch_idx, input_tuple in enumerate(val_generator_eval):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                mrn = list(input_tuple[2])
                # print(mrn_str)
                input_tuple = input_tuple[0:2]
                # print(input_tuple)
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                # Model make prediction
                pred = self.model(input_tensor)
                pred_.extend(torch.argmax(torch.sigmoid(pred), dim=1).cpu().numpy())
                pred2_.extend(np.array(torch.sigmoid(pred).cpu().numpy()).tolist())

                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

                x_val_clinic, y_val_clinic = \
                    clinical_data(args=self.args,
                                  val_mrn_list=mrn,
                                  train_mrn_list=self.args.train_mrn_list).load_val_data_batch(batch_size=1)

                y_pred_val_clinic = clinical_model_.predict_proba(x_val_clinic)
                target2_c.extend(y_val_clinic)
                pred2_c.extend(np.array(y_pred_val_clinic).tolist())

        target2_ = np.array(target2_)
        pred2_ = np.array(pred2_)

        pred2_c_1 = [x[1] for x in pred2_c]

        ### Evaluating loaded data above (Validation, Train, Test)
        # printing precision, recall, and fscore for each class for validation data
        print('\nVALIDATION INFORMATION - IMAGE/CLINIC BASED MODEL\n')

        pred2_[:, 0] = np.add(np.dot(1 - pred2_[:, 1], alpha_image),
                              np.dot([1 - x for x in pred2_c_1], (1 - alpha_image)))

        pred2_[:, 1] = np.add(np.dot(pred2_[:, 1], alpha_image),
                              np.dot(pred2_c_1, (1 - alpha_image)))

        assert [np.argmax(np.array(x).tolist()) for x in target2_] == [np.argmax(x) for x in np.array(target2_c)]

        print('COMBINED IMAGE AND CLINIC MODEL - VALIDATION DATA')

        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report([np.argmax(np.array(x).tolist()) for x in target2_],
                                          np.argmax(pred2_, axis=-1),
                                          target_names=target_names))

        # PRED2_ HAS BEEN UPDATED IN THE CODE ABOVE AND REPRESENTS PREDICTION SCORE OF IMAGE AND CLINIC COMBINED
        fig_add = roc_auc_plot(
            np.array([to_categorical(np.argmax(np.array(x)), self.args.n_classes).tolist() for x in target2_]),
            # target2_c,
            pred2_,
            data_title='Validation Image/Clinic ROC')
        self.tb_logger.add_figure(f"Validation_ROC_Image_Clinic", figure=fig_add)

        # print(np.array([np.argmax(np.array(x)).tolist() for x in target2_]))

        auc_ = roc_auc_score(
            y_true=np.array([np.argmax(np.array(x)).tolist() for x in target2_]),
            y_score=pred2_[:, 1])
        print('combine image and clinic validation auc: ', auc_)

        # Evaluating image and clinic model performance on TEST data

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        target2_c = []
        pred2_c = []

        for batch_idx, input_tuple in enumerate(test_generator_eval):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                mrn = list(input_tuple[2])
                # print(mrn_str)
                input_tuple = input_tuple[0:2]
                # print(input_tuple)
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                # Model make prediction
                pred = self.model(input_tensor)
                pred_.extend(torch.argmax(torch.sigmoid(pred), dim=1).cpu().numpy())
                pred2_.extend(np.array(torch.sigmoid(pred).cpu().numpy()).tolist())

                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

                x_test_clinic, y_test_clinic = \
                    clinical_data(args=self.args,
                                  test_mrn_list=mrn,
                                  train_mrn_list=self.args.train_mrn_list).load_test_data_batch(batch_size=1)

                y_pred_test_clinic = clinical_model_.predict_proba(x_test_clinic)
                target2_c.extend(y_test_clinic)
                pred2_c.extend(np.array(y_pred_test_clinic).tolist())

        target2_ = np.array(target2_)
        pred2_ = np.array(pred2_)

        pred2_c_1 = [x[1] for x in pred2_c]

        ### Evaluating loaded data above (Validation, Train, Test)
        # printing precision, recall, and fscore for each class for validation data
        print('\nTest INFORMATION - IMAGE/CLINIC BASED MODEL\n')

        pred2_[:, 0] = np.add(np.dot(1 - pred2_[:, 1], alpha_image),
                              np.dot([1 - x for x in pred2_c_1], (1 - alpha_image)))

        pred2_[:, 1] = np.add(np.dot(pred2_[:, 1], alpha_image),
                              np.dot(pred2_c_1, (1 - alpha_image)))

        assert [np.argmax(np.array(x).tolist()) for x in target2_] == [np.argmax(x) for x in np.array(target2_c)]

        print('COMBINED IMAGE AND CLINIC MODEL - Test DATA')

        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report([np.argmax(np.array(x).tolist()) for x in target2_],
                                          np.argmax(pred2_, axis=-1),
                                          target_names=target_names))


        # PRED2_ HAS BEEN UPDATED IN THE CODE ABOVE AND REPRESENTS PREDICTION SCORE OF IMAGE AND CLINIC COMBINED
        fig_add = roc_auc_plot(
            np.array([to_categorical(np.argmax(np.array(x)), self.args.n_classes).tolist() for x in target2_]),
            # target2_c,
            pred2_,
            data_title='Test Image/Clinic ROC')
        self.tb_logger.add_figure(f"Test_ROC_Image_Clinic", figure=fig_add)

        auc_ = roc_auc_score(
            y_true=np.array([np.argmax(np.array(x)).tolist() for x in target2_]),
            y_score=pred2_[:, 1])
        print('combine image and clinic test auc: ', auc_)

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
        auc_cum = []
        prauc_cum = []
        spec_cum = []
        sens_cum = []

        log.info('-------------------------------------------------------------------------------------------')

        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            # Gathering input data; prepare_input sends to gpu
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

            # may need to turn on if want to train fully but off for transfer learning
            # input_tensor.requires_grad = True

            # Model make prediction
            pred = self.model(input_tensor)

            # calculating loss and metrics
            loss = self.criterion_pre(pred, target.long().view(-1))

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
                auc = calculate_auc(pred, target)
                prauc = calculate_prauc(pred, target)
                spec, sens = dr_friendly_measures(pred, target)

                # storing loss and metrics
                loss_cum.append(loss.item())
                auc_cum.append(auc)
                prauc_cum.append(prauc)
                spec_cum.append(spec)
                sens_cum.append(sens)

                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    log.info(f"\tBatch {batch_idx + 1} of {len(self.train_data_loader)}")
                    log.info(
                        f"\tLoss: {loss.item()}, PRAUC: {prauc}, AUC: {auc}, Sensitivity: {sens}, Specificity: {spec}")
                    log.info('\t**************************************************************************')
                else:
                    pass

            # self.tb_logger.add_pr_curve('training_PR_curve', target_, pred_, global_step=0)

            # test_str = f"training_loss-{self.args.short_note}"
            self.tb_logger.add_scalar(f"training_loss", loss.item(), self.train_count)
            self.tb_logger.add_scalar(f"training_auc", auc, self.train_count)
            self.tb_logger.add_scalar(f"training_prauc", prauc, self.train_count)
            self.tb_logger.add_scalar(f"training_sensitivity", sens, self.train_count)
            self.tb_logger.add_scalar(f"training_specificity", spec, self.train_count)

            self.tb_logger.flush()

            # train count for tensorboard logging
            self.train_count += 1

            if not self.args.ci_test:
                # save model
                if batch_idx == 0 and (epoch * len(self.train_data_loader)) != 0 and (
                        epoch * len(self.train_data_loader)) % self.save_interval == 0:
                    # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(self.args.save_folder, epoch, batch_idx)
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
            f"Summary-----Loss: {np.round(sum(loss_cum) / len(loss_cum), 4)}, PRAUC: {np.round(np.nanmedian(prauc_cum), 4)}, "
            f"AUC: {np.round(np.nanmedian(auc_cum), 4)},\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tSensitivity: {np.round(np.nanmedian(sens_cum), 4)}, "
            f"Specificity: {np.round(np.nanmedian(spec_cum), 4)}\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
            f"Epoch: {res} (min:seconds), # nan: {sum(math.isnan(x) for x in auc_cum)}")
        log.info('-------------------------------------------------------------------------------------------')

    def validate_epoch_alex(self, epoch):
        self.model.eval()

        # Storing epoch values obtained from batch calculations
        loss_cum = []
        auc_cum = []
        prauc_cum = []
        spec_cum = []
        sens_cum = []

        # starting epoch timer
        epoch_start_time = time.time()

        log.info('-------------------------------------------------------------------------------------------')
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):

            if (batch_idx + 1) % self.print_batch_spacing == 0:
                log.info('*************************************')
                log.info(f"\tBatch {batch_idx + 1} of {len(self.valid_data_loader)}")
            else:
                pass

            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                # input_tensor.requires_grad = False

                pred = self.model(input_tensor)

                loss = self.criterion_pre(pred, target.long().view(-1))
                auc = calculate_auc(pred, target)
                prauc = calculate_prauc(pred, target)
                spec, sens = dr_friendly_measures(pred, target)

                # storing loss and metrics
                loss_cum.append(loss.item())
                auc_cum.append(auc)
                prauc_cum.append(prauc)
                spec_cum.append(spec)
                sens_cum.append(sens)

                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    # log.info('\t**************************************************************************')
                    log.info(f"\tBatch {batch_idx + 1} of {len(self.train_data_loader)}")
                    log.info(
                        f"\tLoss: {loss.item()}, PRAUC: {prauc}, AUC: {auc}, Sensitivity: {sens}, Specificity: {spec}")
                    log.info('\t**************************************************************************')
                else:
                    pass

                self.tb_logger.add_scalar(f"val_loss", loss.item(), self.val_count)
                self.tb_logger.add_scalar(f"val_auc", auc, self.val_count)
                self.tb_logger.add_scalar(f"val_prauc", prauc, self.val_count)
                self.tb_logger.add_scalar(f"val_sensitivity", sens, self.val_count)
                self.tb_logger.add_scalar(f"val_specificity", spec, self.val_count)

                self.tb_logger.flush()

                self.val_count += 1

        self.val_loss = sum(loss_cum) / len(loss_cum)

        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)
        log.info(
            f"Summary-----Loss: {np.round(sum(loss_cum) / len(loss_cum), 4)}, PRAUC: {np.round(np.nanmedian(prauc_cum), 4)}, "
            f"AUC: {np.round(np.nanmedian(auc_cum), 4)},\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tSensitivity: {np.round(np.nanmedian(sens_cum), 4)}, "
            f"Specificity: {np.round(np.nanmedian(spec_cum), 4)}\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
            f"Epoch: {res} (min:seconds), # nan: {sum(math.isnan(x) for x in auc_cum)}")
        log.info('-------------------------------------------------------------------------------------------')
