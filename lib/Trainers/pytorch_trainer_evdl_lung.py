# Python Modules
import numpy as np
import pandas as pd
import os
import time
import math
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score, classification_report
import copy
import ast
from scipy.stats import dirichlet

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
from lib.Loading.lung_cancer_data_loader import Lung_Cancer_Classification

import matplotlib.pyplot as plt

# new metrics from torchmetrics
from torchmetrics import Precision, F1Score

# Evidential ML
from evdl.pytorch_class_uncertainty.losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, \
    relu_evidence
from evdl.pytorch_class_uncertainty.helpers import get_device, one_hot_embedding


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
        outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)

        try:
            tn, fp, fn, tp = confusion_matrix(targets.cpu().numpy(), outputs.cpu().numpy()).ravel()
            specificity = tn / (tn + fp + 1e-12)
            sensitivity = tp / (tp + fn + 1e-12)
            return specificity, sensitivity
        except:
            return np.nan, np.nan


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


def dr_friendly_measures_reg(outputs, targets):
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
        # print(f"loss: {outputs.size()}")
        outputs = F.softmax(outputs, dim=1)
        # print(outputs)
        # print(outputs.size())
        try:
            auc = roc_auc_score(targets.cpu().numpy(), outputs.type(torch.FloatTensor).cpu().data.numpy()[:, 1])
            return auc
        except:
            return np.nan


def calculate_prauc(outputs, targets):
    with torch.no_grad():
        outputs = F.softmax(outputs, dim=1)
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

        self.args.store_outcome_train_values = {'loss': [], 'auc': [], 'pr_auc': [], 'sensitivity': [],
                                                'specificity': [], 'f1': [], 'precision': []}
        self.args.store_outcome_val_values = {'loss': [], 'auc': [], 'pr_auc': [], 'sensitivity': [], 'specificity': [],
                                              'f1': [], 'precision': []}

        self.args.best_model_weights = None

        self.args.best_acc = 0.0
        self.args.no_update_count = 0
        self.args.lr_reset_count = 0

        self.args.epoch_num_best_saved = 0

        for epoch in range(self.start_epoch, (self.args.n_epochs + 1)):

            # self.args.train_running_loss = 0.0
            # self.args.train_running_corrects = 0.0
            # # self.args.train_correct = 0
            #
            # self.args.val_running_loss = 0.0
            # self.args.val_running_corrects = 0.0
            # # self.args.val_correct = 0

            for param_group in self.optimizer.param_groups:
                lr_show = param_group['lr']
            print('\n########################################################################')
            print(f"Training epoch: {epoch}, Learning rate: {lr_show:.8f}")

            self.train_epoch(epoch)

            if self.do_validation:
                print(f"Validation epoch: {epoch}")
                self.validate_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Saving training values (train and val to graph)
        train_val_values_store_dir = './Data/train_loss_eval/'

        train_val_values_model_dir = os.path.join(train_val_values_store_dir, self.args.short_note)
        if not os.path.exists(train_val_values_model_dir):
            os.makedirs(train_val_values_model_dir)
        else:
            pass

        for k, v in self.args.store_outcome_train_values.items():
            plt.plot(v)
            plt.title(k)
            plt.savefig(os.path.join(train_val_values_model_dir, f"{self.args.short_note}_{k}_train_fold_"
                                                                 f"{self.args.single_fold}.jpg"))
            plt.clf()

        for k, v in self.args.store_outcome_val_values.items():
            plt.plot(v)
            plt.title(k)
            plt.savefig(os.path.join(train_val_values_model_dir,
                                     f"{self.args.short_note}_{k}_val_fold_{self.args.single_fold}.jpg"))
            plt.clf()

        # Saving final model after training
        # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
        # after model completion
        if int(self.args.epoch_num_best_saved) >= 10:
            print('using saved model')
            self.model.load_state_dict(self.args.best_model_weights)

            model_save_path = '{}_epoch_{}_{}_fold_{}.pth.tar'.format(self.args.save_folder,
                                                                      int(self.args.epoch_num_best_saved),
                                                                      self.args.short_note,
                                                                      self.args.true_cv_count)

            model_save_dir = os.path.dirname(model_save_path)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            else:
                pass

            print('Save checkpoints: epoch = {}'.format(epoch))
            torch.save({
                'epoch': int(self.args.epoch_num_best_saved),
                # 'batch_id': batch_idx,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()},
                model_save_path)

        else:
            model_save_path = '{}_epoch_{}_{}_fold_{}.pth.tar'.format(self.args.save_folder, epoch,
                                                                      self.args.short_note,
                                                                      self.args.true_cv_count)

            model_save_dir = os.path.dirname(model_save_path)

            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            print('Save checkpoints: epoch = {}'.format(epoch))
            torch.save({
                'epoch': epoch,
                # 'batch_id': batch_idx,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()},
                model_save_path)

        self.args.best_model_weights = 'clearing out'

        ### ASSESSING IMAGE MODEL ###

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        u_ = []
        alpha_ = []
        evidence_ = []

        # cycle through data loader for training data
        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                outputs = self.model(input_tensor)

                # preds is either 0 or 1. so far there are no probabilities output
                _, preds = torch.max(outputs, 1)

                match = torch.reshape(torch.eq(preds, target).float(), (-1, 1))
                evidence = relu_evidence(outputs)
                alpha = evidence + 1

                # u is the metric for uncertainty where 1 is total uncertainty (check paper again)
                u = self.args.n_classes / torch.sum(alpha, dim=1, keepdim=True)

                ## Mike is adding calculations per his interpretation of paper
                S = self.args.n_classes / u
                class_prob_pred = alpha / S

                u_.extend(u.cpu().numpy())
                alpha_.extend(alpha.cpu().numpy())
                evidence_.extend(evidence.cpu().numpy())

                # d_mean_.extend(dirichlet(alpha=alpha.cpu().numpy()).mean())
                # d_var_.extend(dirichlet(alpha=alpha.cpu().numpy()).var())
                ##

                pred_.extend(torch.argmax(class_prob_pred, dim=1).cpu().numpy())
                pred2_.extend(np.array(class_prob_pred.cpu().numpy()).tolist())
                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

        # changed 12/24/2021; testing
        # target2_ = np.array(target2_)
        target2_ = np.array(np.eye(self.args.n_classes)[target_])
        assert all(
            [np.shape(x)[0] == self.args.n_classes for x in
             target2_]), f"make sure one hot predictions is correct: {np.shape(target2_)}"
        pred2_ = np.array(pred2_)

        def select_values_for_cal(value, ind_):
            return value[ind_]

        store_all_label_pred_dir = r'./Data/saved_predictions/evidential_dl'

        if not os.path.exists(store_all_label_pred_dir):
            os.makedirs(store_all_label_pred_dir)
        else:
            pass

        image_train_df = pd.DataFrame()
        image_train_df['true'] = [np.argmax(x) for x in target2_]
        image_train_df['pred'] = [x[1] for x in pred2_]
        image_train_df['pred_final'] = pred_
        image_train_df['alpha'] = alpha_
        image_train_df['u'] = [x[0] for x in u_]
        image_train_df['evidence'] = evidence_
        image_train_df['d_mean'] = [dirichlet(alpha=x).mean() for x in alpha_]
        image_train_df['epistemic'] = image_train_df.apply(
            lambda row: select_values_for_cal(row['d_mean'], row['pred_final']), axis=1)

        image_train_df['d_var'] = [dirichlet(alpha=x).var() for x in alpha_]
        image_train_df['aleatoric'] = image_train_df.apply(
            lambda row: select_values_for_cal(row['d_var'], row['pred_final']), axis=1)

        filename = f"image_model_train_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        image_train_df.to_csv(path_or_buf=csv_dir, index=False)

        # Quick uncertainty estimate
        auc_tm = roc_auc_score(y_true=np.argmax(target2_, axis=-1), y_score=pred2_[:, 1])
        median_unc_train = np.median(u_)

        uncert_threshold = [50, 60, 70, 80]

        for u_thres in uncert_threshold:
            print(f"percentile evaluation {u_thres}")

            df = image_train_df

            try:
                high_uncer = df.loc[df['u'] >= np.percentile(df['u'], u_thres)]
                low_uncer = df.loc[df['u'] < np.percentile(df['u'], u_thres)]
                auc_tm_high = roc_auc_score(y_true=high_uncer['true'], y_score=high_uncer['pred'])
                auc_tm_low = roc_auc_score(y_true=low_uncer['true'], y_score=low_uncer['pred'])

                epistemic_high_med = np.percentile(high_uncer['epistemic'], 50)
                epistemic_high_std = np.std(high_uncer['epistemic'])

                aleatoric_high_med = np.percentile(high_uncer['aleatoric'], 50)
                aleatoric_high_std = np.std(high_uncer['aleatoric'])

                epistemic_low_med = np.percentile(low_uncer['epistemic'], 50)
                epistemic_low_std = np.std(low_uncer['epistemic'])

                aleatoric_low_med = np.percentile(low_uncer['aleatoric'], 50)
                aleatoric_low_std = np.std(low_uncer['aleatoric'])

            except:
                auc_tm_high = np.nan
                auc_tm_low = np.nan

                epistemic_high_med = np.nan
                epistemic_high_std = np.nan

                aleatoric_high_med = np.nan
                aleatoric_high_std = np.nan

                epistemic_low_med = np.nan
                epistemic_low_std = np.nan

                aleatoric_low_med = np.nan
                aleatoric_low_std = np.nan

            print(f"image model train auc: {auc_tm}")
            print(f"image model train auc (high uncert): {auc_tm_high}")
            print(
                f"image model train auc (high uncert - epistemic (med, std): {epistemic_high_med} +/- {epistemic_high_std}")
            print(
                f"image model train auc (high uncert - aleatoric (med, std): {aleatoric_high_med} +/- {aleatoric_high_std}")

            print(f"image model train auc (low uncert): {auc_tm_low}")
            print(
                f"image model train auc (low uncert - epistemic (med, std): {epistemic_low_med} +/- {epistemic_low_std}")
            print(
                f"image model train auc (low uncert - aleatoric (med, std): {aleatoric_low_med} +/- {aleatoric_low_std}")

            print('')
        # new_df = new_df.loc[new_df['mean_entropy'] >= median_val_entropy]

        print('\nTRAIN INFORMATION - image BASED MODEL\n')
        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report(image_train_df['true'],
                                          image_train_df['pred_final'],
                                          target_names=target_names))

        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(target2_, pred2_, data_title='Training ROC')
            self.tb_logger.add_figure(f"Image Model Train AUC", figure=fig_add)

        target_ = torch.from_numpy(np.array(target_).astype('int32'))
        pred_ = torch.from_numpy(np.array(pred_).astype('int32'))

        if self.args.use_tb.lower() == 'true':
            self.tb_logger.add_pr_curve(f"Image Model Training PR Curve", target_, pred_)
            self.tb_logger.flush()

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        alpha_ = []
        u_ = []
        evidence_ = []

        # cycle through data loader for validation data
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                outputs = self.model(input_tensor)

                # preds is either 0 or 1. so far there are no probabilities output
                _, preds = torch.max(outputs, 1)

                match = torch.reshape(torch.eq(preds, target).float(), (-1, 1))
                evidence = relu_evidence(outputs)
                alpha = evidence + 1

                # u is the metric for uncertainty where 1 is total uncertainty (check paper again)
                u = self.args.n_classes / torch.sum(alpha, dim=1, keepdim=True)

                ## Mike is adding calculations per his interpretation of paper
                S = self.args.n_classes / u
                class_prob_pred = alpha / S
                ##

                u_.extend(u.cpu().numpy())
                alpha_.extend(alpha.cpu().numpy())
                evidence_.extend(evidence.cpu().numpy())

                # d_mean_.extend(dirichlet(alpha=alpha.cpu().numpy()).mean())
                # d_var_.extend(dirichlet(alpha=alpha.cpu().numpy()).var())

                # print(f"train: {F.softmax(pred, dim=1).size()}")
                # print(F.softmax(pred, dim=1))

                pred_.extend(torch.argmax(class_prob_pred, dim=1).cpu().numpy())
                pred2_.extend(np.array(class_prob_pred.cpu().numpy()).tolist())
                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())

        # changed 12/24/2021; testing
        # target2_ = np.array(target2_)
        target2_ = np.array(np.eye(self.args.n_classes)[target_])
        # print('val length target: ', np.shape(target2_))
        assert all(
            [np.shape(x)[0] == self.args.n_classes for x in
             target2_]), f"make sure one hot predictions is correct: {np.shape(target2_)}"
        pred2_ = np.array(pred2_)
        # print('val length pred2_: ', np.shape(pred2_))

        # storing predictions for image model
        image_val_df = pd.DataFrame()
        image_val_df['true'] = [np.argmax(x) for x in target2_]
        image_val_df['pred'] = [x[1] for x in pred2_]
        image_val_df['pred_final'] = pred_
        image_val_df['alpha'] = alpha_
        image_val_df['u'] = [x[0] for x in u_]
        image_val_df['evidence'] = evidence_
        image_val_df['d_mean'] = [dirichlet(alpha=x).mean() for x in alpha_]
        image_val_df['epistemic'] = image_val_df.apply(
            lambda row: select_values_for_cal(row['d_mean'], row['pred_final']), axis=1)

        image_val_df['d_var'] = [dirichlet(alpha=x).var() for x in alpha_]
        image_val_df['aleatoric'] = image_val_df.apply(
            lambda row: select_values_for_cal(row['d_var'], row['pred_final']), axis=1)

        filename = f"image_model_val_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        image_val_df.to_csv(path_or_buf=csv_dir, index=False)

        # Quick uncertainty estimate
        # AUC values gathered for total prediction model
        auc_val = roc_auc_score(y_true=np.argmax(target2_, axis=-1), y_score=pred2_[:, 1])

        median_unc_val = np.median(u_)

        store_uncert_cutoff_val = []

        for u_thres in uncert_threshold:
            print(f"percentile evaluation {u_thres}")

            df = image_val_df

            try:
                high_uncer = df.loc[df['u'] >= np.percentile(df['u'], u_thres)]
                low_uncer = df.loc[df['u'] < np.percentile(df['u'], u_thres)]
                auc_tm_high = roc_auc_score(y_true=high_uncer['true'], y_score=high_uncer['pred'])
                auc_tm_low = roc_auc_score(y_true=low_uncer['true'], y_score=low_uncer['pred'])

                store_uncert_cutoff_val.append(np.percentile(df['u'], u_thres))

                epistemic_high_med = np.percentile(high_uncer['epistemic'], 50)
                epistemic_high_std = np.std(high_uncer['epistemic'])

                aleatoric_high_med = np.percentile(high_uncer['aleatoric'], 50)
                aleatoric_high_std = np.std(high_uncer['aleatoric'])

                epistemic_low_med = np.percentile(low_uncer['epistemic'], 50)
                epistemic_low_std = np.std(low_uncer['epistemic'])

                aleatoric_low_med = np.percentile(low_uncer['aleatoric'], 50)
                aleatoric_low_std = np.std(low_uncer['aleatoric'])

            except:
                auc_tm_high = np.nan
                auc_tm_low = np.nan

                epistemic_high_med = np.nan
                epistemic_high_std = np.nan

                aleatoric_high_med = np.nan
                aleatoric_high_std = np.nan

                epistemic_low_med = np.nan
                epistemic_low_std = np.nan

                aleatoric_low_med = np.nan
                aleatoric_low_std = np.nan

            print(f"image model val auc: {auc_val}")
            print(f"image model val auc (high uncert): {auc_tm_high}")
            print(
                f"image model val auc (high uncert - epistemic (med, std): {epistemic_high_med} +/- {epistemic_high_std}")
            print(
                f"image model val auc (high uncert - aleatoric (med, std): {aleatoric_high_med} +/- {aleatoric_high_std}")

            print(f"image model val auc (low uncert): {auc_tm_low}")
            print(
                f"image model val auc (low uncert - epistemic (med, std): {epistemic_low_med} +/- {epistemic_low_std}")
            print(
                f"image model val auc (low uncert - aleatoric (med, std): {aleatoric_low_med} +/- {aleatoric_low_std}")

            print('')

        print('\nVALIDATION INFORMATION - image BASED MODEL\n')
        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report(image_val_df['true'],
                                          image_val_df['pred_final'],
                                          target_names=target_names))

        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(target2_, pred2_, data_title='Validation ROC')
            self.tb_logger.add_figure(f"Image Model Validation AUC", figure=fig_add)

        target_ = torch.from_numpy(np.array(target_).astype('int32'))
        pred_ = torch.from_numpy(np.array(pred_).astype('int32'))

        if self.args.use_tb.lower() == 'true':
            self.tb_logger.add_pr_curve(f"Image Model Validation PR Curve", target_, pred_)
            self.tb_logger.flush()

        target_ = []
        pred_ = []
        target2_ = []
        pred2_ = []

        u_ = []
        alpha_ = []
        evidence_ = []

        # Testing
        # cycle through data loader for testing data
        for batch_idx, input_tuple in enumerate(self.test_data_loader):
            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

                outputs = self.model(input_tensor)

                # preds is either 0 or 1. so far there are no probabilities output
                _, preds = torch.max(outputs, 1)

                match = torch.reshape(torch.eq(preds, target).float(), (-1, 1))
                evidence = relu_evidence(outputs)
                alpha = evidence + 1

                # u is the metric for uncertainty where 1 is total uncertainty (check paper again)
                u = self.args.n_classes / torch.sum(alpha, dim=1, keepdim=True)

                ## Mike is adding calculations per his interpretation of paper
                S = self.args.n_classes / u
                class_prob_pred = alpha / S
                ##

                u_.extend(u.cpu().numpy())
                alpha_.extend(alpha.cpu().numpy())
                evidence_.extend(evidence.cpu().numpy())

                # d_mean_.extend(dirichlet(alpha=alpha.cpu().numpy()).mean())
                # d_var_.extend(dirichlet(alpha=alpha.cpu().numpy()).var())

                # print(f"train: {F.softmax(pred, dim=1).size()}")
                # print(F.softmax(pred, dim=1))
                pred_.extend(torch.argmax(class_prob_pred, dim=1).cpu().numpy())
                pred2_.extend(np.array(class_prob_pred.cpu().numpy()).tolist())
                target_.extend(target.cpu().numpy())
                target2_.extend(np.array(F.one_hot(target).cpu().numpy()).tolist())


        target2_ = np.array(np.eye(self.args.n_classes)[target_])

        assert all(
            [np.shape(x)[0] == self.args.n_classes for x in
             target2_]), f"make sure one hot predictions is correct: {np.shape(target2_)}"
        pred2_ = np.array(pred2_)
        # print('test length pred2_: ', np.shape(pred2_))

        # storing predictions for image model
        image_test_df = pd.DataFrame()
        image_test_df['true'] = [np.argmax(x) for x in target2_]
        image_test_df['pred'] = [x[1] for x in pred2_]
        image_test_df['pred_final'] = pred_
        image_test_df['alpha'] = alpha_
        image_test_df['u'] = [x[0] for x in u_]
        image_test_df['evidence'] = evidence_
        image_test_df['d_mean'] = [dirichlet(alpha=x).mean() for x in alpha_]
        image_test_df['epistemic'] = image_test_df.apply(
            lambda row: select_values_for_cal(row['d_mean'], row['pred_final']), axis=1)

        image_test_df['d_var'] = [dirichlet(alpha=x).var() for x in alpha_]
        image_test_df['aleatoric'] = image_test_df.apply(
            lambda row: select_values_for_cal(row['d_var'], row['pred_final']), axis=1)

        filename = f"image_model_test_prediction_" \
                   f"{self.args.short_note}_{self.args.class_weights}_{self.args.single_fold + 1}_of_{self.args.cv_num}_" \
                   f"seed_{self.args.manual_seed}_{self.args.time_stamp}.csv"
        csv_dir = os.path.join(store_all_label_pred_dir, filename)
        image_test_df.to_csv(path_or_buf=csv_dir, index=False)


        # val uncertainty is being used on purpose
        auc_test = roc_auc_score(y_true=np.argmax(target2_, axis=-1), y_score=pred2_[:, 1])
        # print(f"test auc: {auc_test}")

        for i, u_thres in enumerate(uncert_threshold):
            print(f"percentile evaluation {u_thres}")

            df = image_test_df

            try:
                high_uncer = df.loc[df['u'] >= store_uncert_cutoff_val[i]]
                low_uncer = df.loc[df['u'] < store_uncert_cutoff_val[i]]
                auc_tm_high = roc_auc_score(y_true=high_uncer['true'], y_score=high_uncer['pred'])
                auc_tm_low = roc_auc_score(y_true=low_uncer['true'], y_score=low_uncer['pred'])

                epistemic_high_med = np.percentile(high_uncer['epistemic'], 50)
                epistemic_high_std = np.std(high_uncer['epistemic'])

                aleatoric_high_med = np.percentile(high_uncer['aleatoric'], 50)
                aleatoric_high_std = np.std(high_uncer['aleatoric'])

                epistemic_low_med = np.percentile(low_uncer['epistemic'], 50)
                epistemic_low_std = np.std(low_uncer['epistemic'])

                aleatoric_low_med = np.percentile(low_uncer['aleatoric'], 50)
                aleatoric_low_std = np.std(low_uncer['aleatoric'])

            except:
                auc_tm_high = np.nan
                auc_tm_low = np.nan

                epistemic_high_med = np.nan
                epistemic_high_std = np.nan

                aleatoric_high_med = np.nan
                aleatoric_high_std = np.nan

                epistemic_low_med = np.nan
                epistemic_low_std = np.nan

                aleatoric_low_med = np.nan
                aleatoric_low_std = np.nan

            print(f"image model test auc: {auc_test}")
            print(f"image model test auc (high uncert): {auc_tm_high}")
            print(
                f"image model test auc (high uncert - epistemic (med, std): {epistemic_high_med} +/- {epistemic_high_std}")
            print(
                f"image model test auc (high uncert - aleatoric (med, std): {aleatoric_high_med} +/- {aleatoric_high_std}")

            print(f"image model test auc (low uncert): {auc_tm_low}")
            print(
                f"image model test auc (low uncert - epistemic (med, std): {epistemic_low_med} +/- {epistemic_low_std}")
            print(
                f"image model test auc (low uncert - aleatoric (med, std): {aleatoric_low_med} +/- {aleatoric_low_std}")

            print('')

        print('\nTEST INFORMATION - IMAGE BASED MODEL\n')
        target_names = ['class ' + str(x) for x in range(self.args.n_classes)]
        print('\n', classification_report(image_test_df['true'],
                                          image_test_df['pred_final'],
                                          target_names=target_names))

        if self.args.use_tb.lower() == 'true':
            fig_add = roc_auc_plot(target2_, pred2_, data_title='Test ROC')
            self.tb_logger.add_figure(f"Image Model Test AUC", figure=fig_add)

        target_ = torch.from_numpy(np.array(target_).astype('int32'))
        pred_ = torch.from_numpy(np.array(pred_).astype('int32'))

        if self.args.use_tb.lower() == 'true':
            self.tb_logger.add_pr_curve(f"Image Model Test PR Curve", target_, pred_)
            self.tb_logger.flush()

    def train_epoch(self, epoch):

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
        acc_cum = []
        prauc_cum = []
        spec_cum = []
        sens_cum = []

        f1_cum = []
        prec_cum = []

        youden_cum = []

        print('-------------------------------------------------------------------------------------------')

        for batch_idx, input_tuple in enumerate(self.train_data_loader):

            # need to zero gradient (moved 5/3/2022); just cant be between loss.backward() and optimizer.step()
            self.model.zero_grad()

            # Gathering input data; prepare_input sends to gpu
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

            ## taken from dougbrion
            # need to confirm y variable
            y = one_hot_embedding(target, self.args.n_classes)
            y = y.to(torch.device('cuda'))

            outputs = self.model(input_tensor)

            # preds is either 0 or 1. so far there are no probabilities output
            _, preds = torch.max(outputs, 1)

            match = torch.reshape(torch.eq(preds, target).float(), (-1, 1))
            acc = torch.mean(match)
            evidence = relu_evidence(outputs)
            alpha = evidence + 1

            # u is the metric for uncertainty where 1 is total uncertainty (check paper again)
            u = self.args.n_classes / torch.sum(alpha, dim=1, keepdim=True)

            ## Mike is adding calculations per his interpretation of paper
            S = self.args.n_classes / u
            class_prob_pred = alpha / S
            ##

            loss = edl_log_loss(
                outputs, y.float(), epoch, self.args.n_classes, 10, torch.device('cuda'))

            class_weight_1 = ast.literal_eval(self.args.class_weights)[1]
            class_weight_0 = ast.literal_eval(self.args.class_weights)[0]
            weight = torch.FloatTensor([class_weight_1 if x == 1 else class_weight_0 for x in target.cpu().numpy()])
            # print(f"weight: {weight}")
            # print(f"target: {target}")
            # loss += self.criterion_pre(class_prob_pred[:, 1], target.to(torch.float32))
            # loss_bce = torch.nn.BCELoss(weight=weight).to(torch.device('cuda'))

            # loss += torch.multiply(loss_bce(class_prob_pred[:, 1], target.to(torch.float32)), 3)

            total_evidence = torch.sum(evidence, 1, keepdim=True)
            mean_evidence = torch.mean(total_evidence)
            mean_evidence_succ = torch.sum(
                torch.sum(evidence, 1, keepdim=True) * match
            ) / torch.sum(match + 1e-20)
            mean_evidence_fail = torch.sum(
                torch.sum(evidence, 1, keepdim=True) * (1 - match)
            ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

            loss.backward()

            def clip_gradient(optimizer, grad_clip):
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.clamp_(-grad_clip, grad_clip)

            # clip_gradient(self.optimizer, 2)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            self.optimizer.step()

            # Calculating and appending
            with torch.no_grad():
                auc = calculate_auc(class_prob_pred, target)
                prauc = calculate_prauc(class_prob_pred, target)
                spec, sens = dr_friendly_measures(class_prob_pred, target)

                f1 = F1Score(num_classes=self.args.n_classes).to(torch.device('cuda'))
                f1_value = f1(class_prob_pred, target).cpu().numpy()

                prec = Precision(average='macro', num_classes=self.args.n_classes).to(torch.device('cuda'))
                prec_value = prec(class_prob_pred, target).cpu().numpy()

                # storing loss and metrics
                loss_cum.append(loss.item())
                acc_cum.append(acc)
                auc_cum.append(auc)
                prauc_cum.append(prauc)
                spec_cum.append(spec)
                sens_cum.append(sens)
                # youden_cum.append(spec+self.args.youden_sens_weight*sens-1)

                f1_cum.append(f1_value)
                prec_cum.append(prec_value)

                # statistics (#target.data will need editing)
                # self.args.train_running_loss += loss.item() * input_tensor.size(0)
                # self.args.train_running_corrects += torch.sum(preds == target.data)

                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    print(f"\tBatch {batch_idx + 1} of {len(self.train_data_loader)}")
                    print(
                        f"\tLoss: {loss.item()}, PRAUC: {prauc}, AUC: {auc}, Sensitivity: {sens}, Specificity: {spec}")
                    print(
                        f"\tf1: {f1_value}, Precision: {prec_value}")
                    print('\t**************************************************************************')
                else:
                    pass

            # self.tb_logger.add_pr_curve('training_PR_curve', target_, pred_, global_step=0)

            # test_str = f"training_loss-{self.args.short_note}"
            if self.args.use_tb.lower() == 'true':
                self.tb_logger.add_scalar(f"training_loss", loss.item(), self.train_count)
                self.tb_logger.add_scalar(f"training_auc", auc, self.train_count)
                self.tb_logger.add_scalar(f"training_prauc", prauc, self.train_count)
                self.tb_logger.add_scalar(f"training_sensitivity", sens, self.train_count)
                self.tb_logger.add_scalar(f"training_specificity", spec, self.train_count)

                self.tb_logger.flush()

            # train count for tensorboard logging
            self.train_count += 1

        # Calculating time per epoch
        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)

        # this will likely need editing
        # train_epoch_loss = self.args.train_running_loss / len(self.train_data_loader.dataset)
        # train_epoch_acc = self.args.train_running_corrects.double() / len(self.train_data_loader.dataset)

        # train_prauc = np.nanmean(prauc_cum)
        # train_auc = np.nanmean(auc_cum)
        # train_youden_ = np.nanmean(youden_cum)

        # Calculating time per epoch
        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)

        self.args.store_outcome_train_values['loss'].append(np.nanmean(loss_cum))
        self.args.store_outcome_train_values['auc'].append(np.nanmedian(auc_cum))
        self.args.store_outcome_train_values['pr_auc'].append(np.nanmedian(prauc_cum))
        self.args.store_outcome_train_values['sensitivity'].append(np.nanmedian(sens_cum))
        self.args.store_outcome_train_values['specificity'].append(np.nanmedian(spec_cum))
        self.args.store_outcome_train_values['f1'].append(np.nanmedian(f1_cum))
        self.args.store_outcome_train_values['precision'].append(np.nanmedian(prec_cum))

        print(
            f"Summary-----Loss: {np.round(sum(loss_cum) / len(loss_cum), 4)}, PRAUC: {np.round(np.nanmedian(prauc_cum), 4)}, "
            f"AUC: {np.round(np.nanmedian(auc_cum), 4)},\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tSensitivity: {np.round(np.nanmedian(sens_cum), 4)}, "
            f"Specificity: {np.round(np.nanmedian(spec_cum), 4)}\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
            f"f1: {np.round(np.nanmedian(f1_cum), 4)}, precision: {np.round(np.nanmedian(prec_cum), 4)}\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
            f"Epoch: {res} (min:seconds), # nan: {sum(math.isnan(x) for x in auc_cum)}")
        print(f"Epoch: {res} (min:seconds)")
        print('-------------------------------------------------------------------------------------------')

    def validate_epoch(self, epoch):

        self.model.eval()

        # Storing epoch values obtained from batch calculations
        loss_cum = []
        auc_cum = []
        prauc_cum = []
        spec_cum = []
        sens_cum = []

        f1_cum = []
        prec_cum = []
        # youden_cum = []

        # starting epoch timer
        epoch_start_time = time.time()

        print('-------------------------------------------------------------------------------------------')
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            self.model.zero_grad()

            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                # input_tensor.requires_grad = False

                ## taken from dougbrion
                # need to confirm y variable
                y = one_hot_embedding(target, self.args.n_classes)
                y = y.to(torch.device('cuda'))

                outputs = self.model(input_tensor)
                _, preds = torch.max(outputs, 1)
                loss = edl_log_loss(
                    outputs, y.float(), epoch, self.args.n_classes, 10, torch.device('cuda'))

                match = torch.reshape(torch.eq(preds, target).float(), (-1, 1))
                acc = torch.mean(match)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1
                u = self.args.n_classes / torch.sum(alpha, dim=1, keepdim=True)

                ## Mike is adding calculations per his interpretation of paper
                S = self.args.n_classes / u
                class_prob_pred = alpha / S
                ##

                total_evidence = torch.sum(evidence, 1, keepdim=True)
                mean_evidence = torch.mean(total_evidence)
                mean_evidence_succ = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * match
                ) / torch.sum(match + 1e-20)
                mean_evidence_fail = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * (1 - match)
                ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                # statistics (#target.data will need editing)
                # self.args.val_running_loss += loss.item() * input_tensor.size(0)
                # self.args.val_running_corrects += torch.sum(preds == target.data)

                auc = calculate_auc(class_prob_pred, target)
                prauc = calculate_prauc(class_prob_pred, target)
                spec, sens = dr_friendly_measures(class_prob_pred, target)

                f1 = F1Score(num_classes=self.args.n_classes).to(torch.device('cuda'))
                f1_value = f1(class_prob_pred, target).cpu().numpy()

                prec = Precision(average='macro', num_classes=self.args.n_classes).to(torch.device('cuda'))
                prec_value = prec(class_prob_pred, target).cpu().numpy()

                # storing loss and metrics
                loss_cum.append(loss.item())
                auc_cum.append(auc)
                prauc_cum.append(prauc)
                spec_cum.append(spec)
                sens_cum.append(sens)
                f1_cum.append(f1_value)
                prec_cum.append(prec_value)
                # youden_cum.append(spec + self.args.youden_sens_weight * sens - 1)

                if self.args.use_tb.lower() == 'true':
                    self.tb_logger.add_scalar(f"val_loss", loss.item(), self.val_count)
                    self.tb_logger.add_scalar(f"val_auc", auc, self.val_count)
                    self.tb_logger.add_scalar(f"val_prauc", prauc, self.val_count)
                    self.tb_logger.add_scalar(f"val_sensitivity", sens, self.val_count)
                    self.tb_logger.add_scalar(f"val_specificity", spec, self.val_count)

                    self.tb_logger.flush()

                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    # print('\t**************************************************************************')
                    print(f"\tBatch {batch_idx + 1} of {len(self.train_data_loader)}")
                    print(
                        f"\tLoss: {loss.item()}, PRAUC: {prauc}, AUC: {auc}, Sensitivity: {sens}, Specificity: {spec}")
                    print(
                        f"\tf1: {f1_value}, Precision: {prec_value}")
                    print('\t**************************************************************************')
                else:
                    pass

                self.val_count += 1

        # this will likely need editing
        # val_epoch_loss = self.args.val_running_loss / len(self.valid_data_loader.dataset)
        # val_epoch_acc = self.args.val_running_corrects.double() / len(self.valid_data_loader.dataset)
        # val_prauc = np.nanmean(prauc_cum)
        # val_auc = np.nanmean(auc_cum)
        # youden_ = np.nanmean(youden_cum)

        self.val_loss = sum(loss_cum) / len(loss_cum)

        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)

        self.args.store_outcome_val_values['loss'].append(np.nanmean(loss_cum))
        self.args.store_outcome_val_values['auc'].append(np.nanmedian(auc_cum))
        self.args.store_outcome_val_values['pr_auc'].append(np.nanmedian(prauc_cum))
        self.args.store_outcome_val_values['sensitivity'].append(np.nanmedian(sens_cum))
        self.args.store_outcome_val_values['specificity'].append(np.nanmedian(spec_cum))
        self.args.store_outcome_val_values['f1'].append(np.nanmedian(f1_cum))
        self.args.store_outcome_val_values['precision'].append(np.nanmedian(prec_cum))

        # deep copy the model
        metric_ = np.nanmedian(f1_cum) + np.nanmedian(prec_cum)
        if metric_ >= self.args.best_acc and (np.nanmedian(spec_cum) != 0 and np.nanmedian(sens_cum) != 0 \
                                              and np.nanmedian(auc_cum) >= 0.7 and np.nanmedian(prauc_cum) >= 0.7):
            print('saving new best model - f1+precision')
            self.args.best_acc = metric_
            self.args.best_model_weights = copy.deepcopy(self.model.state_dict())
            self.args.epoch_num_best_saved = epoch
            # self.args.best_model = copy.deepcopy(self.model)
        else:
            self.args.no_update_count += 1
            self.args.lr_reset_count += 1

        if self.args.no_update_count >= 1000:
            self.args.no_update_count = 0
            print('reverting model...')
            self.model.load_state_dict(self.args.best_model_weights)
        else:
            pass

        print(
            f"Summary-----Loss: {np.round(sum(loss_cum) / len(loss_cum), 4)}, PRAUC: {np.round(np.nanmedian(prauc_cum), 4)}, "
            f"AUC: {np.round(np.nanmedian(auc_cum), 4)},\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tSensitivity: {np.round(np.nanmedian(sens_cum), 4)}, "
            f"Specificity: {np.round(np.nanmedian(spec_cum), 4)}\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
            f"f1: {np.round(np.nanmedian(f1_cum), 4)}, precision: {np.round(np.nanmedian(prec_cum), 4)}\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"
            f"Epoch: {res} (min:seconds), # nan: {sum(math.isnan(x) for x in auc_cum)}")
        print('-------------------------------------------------------------------------------------------')
