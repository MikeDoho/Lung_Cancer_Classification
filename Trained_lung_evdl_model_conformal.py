# Python libraries
import os
import datetime
import time
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys
import copy

# Pytorch
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import logging

# Lib files
import lib.utils as utils
import lib.Loading as medical_loaders
from lib.utils.setting import parse_opts
from lib.utils.model_single_input_extend_dropout import generate_model
# from lib.utils.model_spatial import generate_model
from lib.Loading.lung_cancer_data_loader import Lung_Cancer_Classification
# import lib.Trainers.pytorch_trainer_pretrain_clinic as pytorch_trainer
import lib.Trainers.pytorch_trainer_pretrain_clinic_softmax_lung as pytorch_trainer
from lib.utils.logger import log
from lib.medzoo.ResNet3DMedNet import generate_resnet3d
from lib.Models.resnet_fork_single_input_extend_dropout import resnet50

# try to address speed issues?
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

# Evidential ML
from evdl.pytorch_class_uncertainty.losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, \
    relu_evidence

cudnn.benchmark = True
cudnn.deterministic = True

time_stamp = datetime.datetime.now()
print("Time stamp " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'), '\n\n')

print("Arguments Used")
args = parse_opts()
args.time_stamp = time_stamp.strftime('%Y.%m.%d')

if float(args.resnet_lr_factor) > 0 and args.batch_size > 16:
    print('\ndownsizing batch size because now we are training the whole model\n')
    args.batch_size = 16
    print(args.batch_size)
else:
    pass

print(f"Setting seed for reproducibility\n\tseed: {args.manual_seed}")
utils.general.reproducibility(args, args.manual_seed)
# print(f"Creating saved folder: {args.save}")
# utils.general.make_dirs(args.save)

print("\nCreating custom training and validation generators")
print(f"\tIs data augmentation being utilized?: {args.augmentation}")
# args.augmentation = ast.literal_eval(args.augmentation)

print(f"\tBatch size: {args.batch_size}")
print(f"selected clinical features: {args.selected_clinical_features}")

# getting data
args.phase = 'train'
if args.no_cuda:
    args.pin_memory = False
else:
    args.pin_memory = True

### ADDED
# Loading csv file that contains list of MRN
if args.exclude_mrn.lower() == 'true':
    exclude_dir = args.exclude_mrn_path
    exclude_csv_filename = args.exclude_mrn_filename
    exclude_mrns = pd.read_csv(os.path.join(exclude_dir, exclude_csv_filename))

    # Currently loaded as dataframe
    exclude_mrns = exclude_mrns['MRN'].tolist()
    exclude_mrns = [str(int(x)) for x in exclude_mrns]
elif args.exclude_mrn.lower() == 'false':
    exclude_mrns = []

args.exclude_mrns = exclude_mrns

# Load Model
model, parameters = generate_model(args)
summary(model, (args.in_modality, args.input_D, args.input_H, args.input_W))

model_to_load = '20220614_res101ext-sm-do005-evdl_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[150]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV'

checkpoint_path_main = '/home/s185479/Python/Working_Projects/Lung_Cancer_Classification/trails/' + \
                       model_to_load

saved_end_model_120 = [
    'resnet_101_epoch_150_20220614_res101ext-sm-do005-evdl_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[150]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV_fold_1.pth.tar',
    'resnet_101_epoch_133_20220614_res101ext-sm-do005-evdl_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[150]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV_fold_2.pth.tar',
    'resnet_101_epoch_130_20220614_res101ext-sm-do005-evdl_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[150]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV_fold_3.pth.tar',
    'resnet_101_epoch_130_20220614_res101ext-sm-do005-evdl_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[150]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV_fold_4.pth.tar',
    'resnet_101_epoch_144_20220614_res101ext-sm-do005-evdl_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[150]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV_fold_5.pth.tar'
    ]

print('saved model: ', saved_end_model_120)

# nonconformist
nonconform_on = True

significance_list = [0.05, 0.1]
cut_off_percent_list = list(np.arange(30, 90, 1))

# personal
personal_on = False
# class_threshold = [0.10, 0.10]
# update conformist pred
adapted_conform_pred = False
alpha = None

# creating dictionary to store
dict_ = {'nonconf': {}}

for i in range(len(saved_end_model_120)):
    dict_['nonconf'].update({f"model_{i}": {}})
    for s in significance_list:
        for c in cut_off_percent_list:
            dict_['nonconf'][f"model_{i}"].update(
                {f"s_{s}_c_{c}": {'total_pred_df': None, 'h_conf_df': None, 'h_cred_df': None,
                                  'l_cred_df': None, 'l_conf_df': None, 'total_AUC': None,
                                  'h_AUC': None, 'l_AUC': None, 'h_cred_AUC': None, 'l_cred_AUC': None,
                                  'h_conf_cred_AUC': None, 'l_conf_cred_AUC': None,
                                  'single_AUC': None, 'double_AUC': None, 'null_AUC': None}})

dict_personal = {'conf_personal': {}}

for i in range(len(saved_end_model_120)):
    dict_personal['conf_personal'].update({f"model_{i}": {}})
    for s in significance_list:
        for c in cut_off_percent_list:
            dict_personal['conf_personal'][f"model_{i}"].update(
                {f"s_{s}_c_{c}": {'total_pred_df': None, 'total_AUC': None,
                                  'single_AUC': None, 'double_AUC': None, 'null_AUC': None
                                  }})

for j, saved_model_name in enumerate(saved_end_model_120):
    print(f"\nCurrent model\n{saved_model_name}\n\n")



    assert saved_model_name.split('.')[-3].split('_')[-1] == str(j + 1), \
        f"need to make sure models are loaded by sequential fold; {saved_model_name.split('.')[-3].split('_')[-1]} and {j + 1}"

    checkpoint = torch.load(os.path.join(checkpoint_path_main, saved_model_name))
    # print(checkpoint['state_dict'])

    net_dict = checkpoint['state_dict']

    pretrain = torch.load(args.pretrain_path)
    # print({k: v for k, v in pretrain['state_dict'].items() if k == 'module.layer1.0.downsample.1.weight'})

    need_to_have = []
    not_to_have = []

    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in need_to_have}
    net_dict.update(pretrain_dict)

    for x in not_to_have:
        net_dict.pop(x)

    model.load_state_dict(net_dict)

    # LOADING DATA (MRN, LABEL) TO LATER PERFORM DATA SPLIT FOR CROSS-VALIDATION
    # train_label path, val_label_path, and test_label_path are all the same
    total_mrn_list = [x.split('/')[-1] for x in os.listdir(args.main_input_dir) if \
                      x.split('/')[-1].split('-')[0] == 'input']

    total_label_list = [int(float(x.split('-')[-1].split('.npy')[0])) for x in total_mrn_list if \
                        x.split('-')[0] == 'input']

    total_outcome_zip = list(zip(total_mrn_list, total_label_list))

    print(f"number of CV: {args.cv_num}")
    skf = StratifiedKFold(n_splits=args.cv_num, random_state=args.manual_seed, shuffle=True)

    # total_mrn_array = np.array(total_mrn_list)
    total_mrn_outcome_list = total_mrn_list
    total_label_array = np.array(total_label_list)

    ###
    import random

    args.random_value_for_fold = [random.randint(0, 10000000) for _ in range(args.cv_num)]
    print(f"Randomly generated numbers associated with folds\n{args.random_value_for_fold}")

    total_mrn_label_list = total_label_list
    # need to stratify based on mrn based outcomes
    stratify_label = total_mrn_label_list
    print(f"initial stratify list: {stratify_label}")

    assert len(total_mrn_outcome_list) == len(total_mrn_label_list), \
        f"confirm input and label length are the same; input: {len(total_mrn_outcome_list)}, label: {len(total_mrn_label_list)}"

    # Data stratification
    args.true_cv_count = 0
    args.cv_count_csv_save = 0

    total_mrn_array = np.array(total_mrn_list)
    stratify_label = np.array(stratify_label)

    for train_index, test_index in skf.split(total_mrn_array, stratify_label):
        #         print("TRAIN:", train_index, "TEST:", test_index)

        X_train, xtest_ = total_mrn_array[train_index], total_mrn_array[test_index]
        y_train, ytest_ = total_label_array[train_index], total_label_array[test_index]

        X_train = list(X_train)
        xtest_ = list(xtest_)
        y_train = list(y_train)
        ytest_ = list(ytest_)

        ### Creating the Split for training and validation and test
        args.single_fold = args.true_cv_count

        # train_test_split_fraction = 0.2
        train_val_split_fraction = 0.25

        xtrain_, xval_, ytrain_, yval_ = train_test_split(X_train, y_train,
                                                          test_size=train_val_split_fraction,
                                                          random_state=args.manual_seed + args.true_cv_count,
                                                          stratify=y_train)

        # print(f"train id: {xtrain_}")
        print(f"general train ratio: {np.round(np.sum(ytrain_) / len(ytrain_), 3)}\n")
        print(f"Number of p_ids with lung cancer in training data: {np.sum(ytrain_)}")
        # print(f"validation id: {xval_}")
        print(f"general validation ratio: {np.round(np.sum(yval_) / len(yval_), 3)}\n")
        print(f"Number of p_ids with lung cancer in val data: {np.sum(yval_)}")
        # print(f"test id: {xtest_}")
        print(f"general testing ratio: {np.round(np.sum(ytest_) / len(ytest_), 3)}\n")
        print(f"Number of p_ids with lung cancer in test data: {np.sum(ytest_)}")

        # Need to take all lymph nodes associated with the MRNs obtained above
        args.train_lung_list = [x for x in total_mrn_outcome_list if x.split('_')[0] in xtrain_ and \
                                ('1' in x.split('_')[-1] or '0' in x.split('_')[-1]) and \
                                x.split('_')[0] not in args.exclude_mrns]

        args.val_lung_list = [x for x in total_mrn_outcome_list if x.split('_')[0] in xval_ and \
                              ('1' in x.split('_')[-1] or '0' in x.split('_')[-1]) and \
                              x.split('_')[0] not in args.exclude_mrns]

        args.test_lung_list = [x for x in total_mrn_outcome_list if x.split('_')[0] in xtest_ and \
                               ('1' in x.split('_')[-1] or '0' in x.split('_')[-1]) and \
                               x.split('_')[0] not in args.exclude_mrns]

        # obtain min and max for normalization

        args.input_min = 0
        args.input_max = 0

        for train_id in args.train_lung_list:
            t_ = np.load(os.path.join(args.main_input_dir, train_id))

            ct_min = np.min(t_)
            ct_max = np.max(t_)

            if ct_min < args.input_min:
                args.input_min = ct_min
            else:
                pass
            if ct_max > args.input_max:
                args.input_max = ct_max
            else:
                pass

        ### ADDED
        # Loading csv file that contains list of MRN
        if args.exclude_mrn.lower() == 'true':
            exclude_dir = args.exclude_mrn_path
            exclude_csv_filename = args.exclude_mrn_filename
            exclude_mrns = pd.read_csv(os.path.join(exclude_dir, exclude_csv_filename))

            # Currently loaded as dataframe
            exclude_mrns = exclude_mrns['MRN'].tolist()
            exclude_mrns = [str(int(x)) for x in exclude_mrns]
        elif args.exclude_mrn.lower() == 'false':
            exclude_mrns = []

        # # Repeated to exclude mrns obtained above
        args.train_mrn_list = [x for x in xtrain_]
        args.val_mrn_list = [x for x in xval_]
        args.test_mrn_list = [x for x in xtest_]

        # adding sampler to training dataloader because of significant class imbalance
        # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/10
        # print('look here', args.weighted_sampler_on, type(args.weighted_sampler_on))
        if args.weighted_sampler_on:
            print('Creating training sampler for class imbalance')
            # target code is the same that is used in the data loading file
            targets = [int(float(x.split('-')[-1].split('.npy')[0])) for x in args.train_lung_list]
            class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
            print("class_sample_count", class_sample_count.shape, class_sample_count)

            weights = 1. / class_sample_count
            weights = np.multiply(weights, ast.literal_eval(args.weighted_sampler_weight_adjust))
            print('adjusted weights by ', str(args.weighted_sampler_weight_adjust))

            # https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/2
            samples_weights = weights[targets]
            print("samples_weights", samples_weights.shape, samples_weights[0:10], '\n')
            assert len(samples_weights) == len(targets)

            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights,
                                                                     len(samples_weights),
                                                                     replacement=True)

            print(
                'Creating validation sampler for less errors displayed in log file; more permanent solution needed')
            # target code is the same that is used in the data loading file
            targets_val = [int(float(x.split('-')[-1].split('.npy')[0])) for x in args.val_lung_list]
            class_sample_count_val = np.array([len(np.where(targets_val == t)[0]) for t in np.unique(targets_val)])
            print("class_sample_count", class_sample_count_val.shape, class_sample_count_val)

            weights_val = 1. / class_sample_count_val
            weights_val = np.multiply(weights_val, ast.literal_eval(args.weighted_sampler_weight_adjust))
            print('adjusted weights by ', str(args.weighted_sampler_weight_adjust))

            # https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/2
            samples_weights_val = weights_val[targets_val]
            print("samples_weights", samples_weights_val.shape, samples_weights_val[0:10], '\n')
            assert len(samples_weights_val) == len(targets_val)

            sampler_val = torch.utils.data.sampler.WeightedRandomSampler(samples_weights_val,
                                                                         len(samples_weights_val),
                                                                         replacement=True)
            shuffle_ = False
        else:
            print('No weighted sampler used')
            sampler = None
            sampler_val = None
            shuffle_ = True

    args.true_cv_count += 1
    # Creating new validation and testing generators for validation
    # VALIDATION
    args.aug_percent = 0.0

    train_dataset_eval = Lung_Cancer_Classification(args,
                                                         mode='train',
                                                         dataset_path=args.main_input_dir,
                                                         exclude_mrns=exclude_mrns,
                                                         clinic_image_eval=False)
    train_generator_eval = DataLoader(train_dataset_eval,
                                    batch_size=1,
                                    sampler=None,
                                    shuffle=True,  # shuffle cannot be on with a sampler
                                    num_workers=0)

    validation_dataset_eval = Lung_Cancer_Classification(args,
                                                    mode='val',
                                                    dataset_path=args.main_input_dir,
                                                    exclude_mrns=exclude_mrns,
                                                    clinic_image_eval=False)
    val_generator_eval = DataLoader(validation_dataset_eval,
                               batch_size=1,
                               sampler=None,
                               shuffle=True,  # shuffle cannot be on with a sampler
                               num_workers=0)

    # TESTING
    test_dataset_eval = Lung_Cancer_Classification(args,
                                                    mode='test',
                                                    dataset_path=args.main_input_dir,
                                                    exclude_mrns=exclude_mrns,
                                                    clinic_image_eval=False)
    test_generator_eval = DataLoader(test_dataset_eval,
                               batch_size=1,
                               sampler=None,
                               shuffle=True,  # shuffle cannot be on with a sampler
                               num_workers=0)


    if nonconform_on:
        # Using Nonconformist library
        from nonconformist.base import ClassifierAdapter
        from nonconformist.nc import ClassifierNc
        from nonconformist.cp import IcpClassifier
        from nonconformist.nc import NcFactory, MarginErrFunc


        class MyClassifierAdapter(ClassifierAdapter):
            def __init__(self, model, fit_params=None):
                super(MyClassifierAdapter, self).__init__(model, fit_params)

            def fit(self, x, y):
                '''
                    x is a numpy.array of shape (n_train, n_features)
                    y is a numpy.array of shape (n_train)

                    Here, do what is necessary to train the underlying model
                    using the supplied training data
                '''



                pass

            def predict(self, x):
                '''
                    Obtain predictions from the underlying model

                    Make sure this function returns an output that is compatible with
                    the nonconformity function used. For default nonconformity functions,
                    output from this function should be class probability estimates in
                    a numpy.array of shape (n_test, n_classes)
                '''

                import torch
                pred_con = []
                pred2_con = []
                # for i, (x, target) in enumerate(fit_params['generator']):
                for i in x:
                    self.model.eval()
                    # target = target.cuda()
                    i = np.expand_dims(i, axis=0)
                    outputs = self.model(torch.Tensor(i).cuda())

                    # outputs = model(image)

                    # preds is either 0 or 1. so far there are no probabilities output
                    _, preds = torch.max(outputs, 1)
                    evidence = relu_evidence(outputs)
                    alpha = evidence + 1

                    # u is the metric for uncertainty where 1 is total uncertainty (check paper again)
                    u = n_classes / torch.sum(alpha, dim=1, keepdim=True)

                    ## Mike is adding calculations per his interpretation of paper
                    S = n_classes / u
                    class_prob_pred = alpha / S

                    outputs = class_prob_pred

                    # pred_con.extend(torch.argmax(F.softmax(pred, dim=1), dim=1).cpu().numpy())
                    pred2_con.extend(np.array(outputs.cpu().detach().numpy()).tolist())

                pred2_con = np.array(pred2_con)
                return np.array(pred2_con)


        ##### USING NONCONFORMIST LIBRARY
        # trying to attempt nonconformal readme https://github.com/donlnz/nonconformist/blob/master/README.ipynb
        nonconform_model = MyClassifierAdapter(model)
        nc = ClassifierNc(nonconform_model, MarginErrFunc())
        icp = IcpClassifier(nc)  # Create an inductive conformal classifier
        icp_t = IcpClassifier(nc)  # Create an inductive conformal classifier
        # Fit the ICP using the proper training set
        # icp.fit(iris.data[idx_train, :], iris.target[idx_train])

        # Calibrate the ICP using the calibration set
        x_val_con = np.zeros((len(args.val_mrn_list), 1, 32, 32, 32))
        y_val_con = []
        pred_con_val = []
        pred2_con_val = []

        for i, (x, target) in enumerate(val_generator_eval):
            # print('individual: ', np.shape(x))

            if i < len(args.val_mrn_list):
                x_val_con[i, ...] = x
                y_val_con.append(target.cpu().numpy())

                model.eval()
                outputs = model(x.cuda())

                # preds is either 0 or 1. so far there are no probabilities output
                _, preds = torch.max(outputs, 1)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1

                # u is the metric for uncertainty where 1 is total uncertainty (check paper again)
                u = n_classes / torch.sum(alpha, dim=1, keepdim=True)

                ## Mike is adding calculations per his interpretation of paper
                S = n_classes / u
                class_prob_pred = alpha / S

                outputs = class_prob_pred
                pred2_con_val.extend(np.array(outputs.cpu().detach().numpy()).tolist())

        pred2_con_val = np.array(pred2_con_val)
        pred2_cont_val_class_1 = [x[1] for x in pred2_con_val]

        x_val_con = np.array(x_val_con)
        y_val_con = np.array(y_val_con)
        icp.calibrate(x_val_con, y_val_con)

        # Calibrate the ICP_T using the train set (looking specifically I
        x_train_con = np.zeros((len(args.train_mrn_list), 1, 32, 32, 32))
        y_train_con = []
        # print('what is the shape: ', np.shape(x_train_con))

        for i, (x, target) in enumerate(train_generator_eval):
            # print('individual: ', np.shape(x))

            if i < len(args.train_mrn_list):
                x_train_con[i, ...] = x
                y_train_con.append(target.cpu().numpy())

        x_train_con = np.array(x_train_con)
        y_train_con = np.array(y_train_con)
        icp_t.calibrate(x_train_con, y_train_con)

        # Produce predictions for the test set, with confidence alpha
        # obtain test set (lazy)
        x_test_con = np.zeros((len(args.test_mrn_list), 1, 32, 32, 32))
        y_test_con = []

        pred_con_test = []
        pred2_con_test = []
        for i, (x, target) in enumerate(test_generator_eval):

            if i < len(args.test_mrn_list):
                x_test_con[i, ...] = x
                y_test_con.append(target.cpu().numpy())

                model.eval()
                outputs = model(x.cuda())

                # preds is either 0 or 1. so far there are no probabilities output
                _, preds = torch.max(outputs, 1)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1

                # u is the metric for uncertainty where 1 is total uncertainty (check paper again)
                u = n_classes / torch.sum(alpha, dim=1, keepdim=True)

                ## Mike is adding calculations per his interpretation of paper
                S = n_classes / u
                class_prob_pred = alpha / S

                outputs = class_prob_pred
                pred2_con_test.extend(np.array(outputs.cpu().detach().numpy()).tolist())

        pred2_con_test = np.array(pred2_con_test)
        pred2_cont_test_class_1 = [x[1] for x in pred2_con_test]

        x_test_con = np.array(x_test_con)
        y_test_con = np.array(y_test_con)

        # significance, cut off
        for signif in significance_list:
            for cut_o in cut_off_percent_list:
                print(f"signif_{signif}_cut_{cut_o}")

                main_df_save_dir = './Data/conformal_prediction_outcomes/'

                if not os.path.exists(main_df_save_dir):
                    os.makedirs(main_df_save_dir)
                else:
                    pass

                significance_save_dir = os.path.join(main_df_save_dir, f"signif_{signif}")

                # creating folder to save predictions
                if not os.path.exists(significance_save_dir):
                    os.makedirs(significance_save_dir)
                else:
                    pass

                # test samples
                prediction = icp.predict(x_test_con, significance=signif)

                print('predict confidence w/ significance: ', signif)
                nonconf_confid_df = pd.DataFrame(icp.predict_conf(x_test_con),
                                                 columns=['Label', 'Confidence', 'Credibility'])
                nonconf_confid_df['class_1_pred'] = pred2_cont_test_class_1
                nonconf_confid_df['y_true'] = y_test_con

                save_csv_dir = r'./Data/conformal_prediction_outcomes/csv_files/'

                if not os.path.exists(save_csv_dir):
                    os.makedirs(save_csv_dir)
                else:
                    pass

                csv_test_filename = f"{saved_model_name}_signif_{signif}_test.csv"
                nonconf_confid_df.to_csv(os.path.join(save_csv_dir, csv_test_filename), index=False)

                complete_test_results = list(zip(pred2_cont_test_class_1, y_test_con, prediction))

                # validation samples
                prediction_val = icp.predict(x_val_con, significance=signif)
                nonconf_confid_val_df = pd.DataFrame(icp.predict_conf(x_val_con),
                                                     columns=['Label', 'Confidence', 'Credibility'])
                nonconf_confid_val_df['class_1_pred'] = pred2_cont_val_class_1
                nonconf_confid_val_df['y_true'] = y_val_con

                csv_val_filename = f"{saved_model_name}_signif_{signif}_val.csv"
                nonconf_confid_val_df.to_csv(os.path.join(save_csv_dir, csv_val_filename), index=False)

                f_f = []
                f_t = []
                t_t = []

                for predc1, targetc1, con_setc1 in complete_test_results:

                    if all([i == False for i in con_setc1]):
                        f_f.append((predc1, targetc1, con_setc1))
                    elif all([i == True for i in con_setc1]):
                        t_t.append((predc1, targetc1, con_setc1))
                    else:
                        f_t.append((predc1, targetc1, con_setc1))

                print('\nAUC nonconformist analyses')
                from sklearn.metrics import roc_auc_score

                # print('null: ', f_f)
                # print('single: ', f_t)
                # print('double: ', t_t)
                #
                if f_t:
                    try:
                        print(
                            f"single pred {len(f_t)}: {roc_auc_score(y_true=[x[1] for x in f_t], y_score=[x[0] for x in f_t])}")
                        dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['single_AUC'] = (
                        roc_auc_score(y_true=[x[1] for x in f_t],
                                      y_score=[x[0] for x in f_t]), len(f_t))
                    except:
                        print(f"single pred: nan")
                        dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['single_AUC'] = (np.nan, 0)

                if t_t:
                    try:
                        print(
                            f"double pred {len(t_t)}: {roc_auc_score(y_true=[x[1] for x in t_t], y_score=[x[0] for x in t_t])}")
                        dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['double_AUC'] = (
                        roc_auc_score(y_true=[x[1] for x in t_t],
                                      y_score=[x[0] for x in t_t]), len(t_t))
                    except:
                        print(f"double pred: nan")
                        dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['double_AUC'] = (np.nan, 0)

                if f_f:
                    try:
                        print(
                            f"null pred {len(f_f)}: {roc_auc_score(y_true=[x[1] for x in f_f], y_score=[x[0] for x in f_f])}")
                        dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['null_AUC'] = (
                        roc_auc_score(y_true=[x[1] for x in f_f],
                                      y_score=[x[0] for x in f_f]), len(f_f))
                    except:
                        print(f"null pred: nan")
                        dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['null_AUC'] = (np.nan, 0)

                print(f"p value predictions full")
                print(f"total {len(nonconf_confid_df)}: "
                      f"{roc_auc_score(y_true=nonconf_confid_df['y_true'], y_score=nonconf_confid_df['class_1_pred'])}")

                # send prediction to csv file
                # nonconf_confid_df.to_csv(os.path.join(significance_save_dir, f"total_df_cutoff[{cut_o}]_sign[{signif}]_model_{j}.csv"))
                header = nonconf_confid_df.columns.tolist()
                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['total_pred_df'] = np.array(nonconf_confid_df)
                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['header'] = header

                confid_cutoff = np.percentile(nonconf_confid_df['Confidence'], cut_o)
                cred_cutoff = np.percentile(nonconf_confid_df['Credibility'], cut_o)

                confid_val_cutoff = np.percentile(nonconf_confid_val_df['Confidence'], cut_o)
                cred_val_cutoff = np.percentile(nonconf_confid_val_df['Credibility'], cut_o)

                # testing just using % as a cut off
                confid_cutoff = cut_o / 100
                cred_cutoff = cut_o / 100

                print('credibility cutoff: ', cred_cutoff)
                print('confidence cutoff: ', confid_cutoff, '\n')

                # obtain split for confidence and credibility (testing data)
                # confidence
                df_high_conf = nonconf_confid_df.loc[nonconf_confid_df.Confidence >= confid_cutoff]
                df_low_conf = nonconf_confid_df.loc[nonconf_confid_df.Confidence < confid_cutoff]

                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_conf_df'] = np.array(df_high_conf)
                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_conf_df'] = np.array(df_low_conf)

                # credibility
                df_high_cred = nonconf_confid_df.loc[nonconf_confid_df.Credibility >= cred_cutoff]
                df_low_cred = nonconf_confid_df.loc[nonconf_confid_df.Credibility < cred_cutoff]

                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_cred_df'] = np.array(df_high_cred)
                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_cred_df'] = np.array(df_low_cred)

                # Confidence and cred
                df_high_conf_cred = nonconf_confid_df.loc[
                    (nonconf_confid_df.Credibility >= cred_cutoff) & (nonconf_confid_df.Confidence >= confid_cutoff)]
                df_low_conf_cred = nonconf_confid_df.loc[
                    (nonconf_confid_df.Credibility < cred_cutoff) & (nonconf_confid_df.Confidence < confid_cutoff)]

                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_conf_cred_df'] = np.array(df_high_conf_cred)
                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_conf_cred_df'] = np.array(df_low_conf_cred)

                # validation
                # obtain split for confidence and credibility
                # confidence
                df_high_val_conf = nonconf_confid_val_df.loc[nonconf_confid_val_df.Confidence >= confid_cutoff]
                df_low_val_conf = nonconf_confid_val_df.loc[nonconf_confid_val_df.Confidence < confid_cutoff]

                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_conf_val_df'] = np.array(df_high_val_conf)
                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_conf_val_df'] = np.array(df_low_val_conf)

                # credibility
                df_high_val_cred = nonconf_confid_val_df.loc[nonconf_confid_val_df.Credibility >= cred_cutoff]
                df_low_val_cred = nonconf_confid_val_df.loc[nonconf_confid_val_df.Credibility < cred_cutoff]

                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_cred_val_df'] = np.array(df_high_val_cred)
                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_cred_val_df'] = np.array(df_low_val_cred)

                # Confidence and cred
                df_high_val_conf_cred = nonconf_confid_val_df.loc[
                    (nonconf_confid_val_df.Credibility >= cred_cutoff) & (
                                nonconf_confid_val_df.Confidence >= confid_cutoff)]
                df_low_val_conf_cred = nonconf_confid_val_df.loc[
                    (nonconf_confid_val_df.Credibility < cred_cutoff) & (
                                nonconf_confid_val_df.Confidence < confid_cutoff)]

                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_conf_cred_val_df'] = np.array(
                    df_high_val_conf_cred)
                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_conf_cred_val_df'] = np.array(
                    df_low_val_conf_cred)

                # testing data
                # confidence
                try:
                    print(f"high confid {len(df_high_conf)}: "
                          f"{roc_auc_score(y_true=df_high_conf['y_true'], y_score=df_high_conf['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_AUC'] = roc_auc_score(
                        y_true=df_high_conf['y_true'],
                        y_score=df_high_conf['class_1_pred'])
                except ValueError:
                    print(f"high confid {len(df_high_conf)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_AUC'] = np.nan

                try:
                    print(f"low confid {len(df_low_conf)}: "
                          f"{roc_auc_score(y_true=df_low_conf['y_true'], y_score=df_low_conf['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_AUC'] = roc_auc_score(
                        y_true=df_low_conf['y_true'],
                        y_score=df_low_conf['class_1_pred'])
                except ValueError:
                    print(f"low confid {len(df_low_conf)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_AUC'] = np.nan

                # credibility
                try:
                    print(f"high cred {len(df_high_cred)}: "
                          f"{roc_auc_score(y_true=df_high_cred['y_true'], y_score=df_high_cred['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_cred_AUC'] = roc_auc_score(
                        y_true=df_high_cred['y_true'],
                        y_score=df_high_cred['class_1_pred'])
                except ValueError:
                    print(f"high cred {len(df_high_cred)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_cred_AUC'] = np.nan

                try:
                    print(f"low cred {len(df_low_cred)}: "
                          f"{roc_auc_score(y_true=df_low_cred['y_true'], y_score=df_low_cred['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_cred_AUC'] = roc_auc_score(
                        y_true=df_low_cred['y_true'],
                        y_score=df_low_cred['class_1_pred'])
                except ValueError:
                    print(f"low cred {len(df_low_cred)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_cred_AUC'] = np.nan

                # confidence and credibility
                try:
                    print(f"high cred {len(df_high_conf_cred)}: "
                          f"{roc_auc_score(y_true=df_high_conf_cred['y_true'], y_score=df_high_conf_cred['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_conf_cred_AUC'] = roc_auc_score(
                        y_true=df_high_conf_cred['y_true'],
                        y_score=df_high_conf_cred['class_1_pred'])
                except ValueError:
                    print(f"high cred {len(df_high_conf_cred)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_conf_cred_AUC'] = np.nan

                try:
                    print(f"low cred {len(df_low_conf_cred)}: "
                          f"{roc_auc_score(y_true=df_low_conf_cred['y_true'], y_score=df_low_conf_cred['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_conf_cred_AUC'] = roc_auc_score(
                        y_true=df_low_conf_cred['y_true'],
                        y_score=df_low_conf_cred['class_1_pred'])
                except ValueError:
                    print(f"low cred {len(df_low_conf_cred)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_conf_cred_AUC'] = np.nan

                store_true_check = df_high_conf['y_true'].tolist()
                store_true_check.extend(df_low_conf['y_true'].tolist())
                store_pred_check = df_high_conf['class_1_pred'].tolist()
                store_pred_check.extend(df_low_conf['class_1_pred'].tolist())
                print(f"total performance {len(store_pred_check)}: "
                      f"{roc_auc_score(y_true=store_true_check, y_score=store_pred_check)}")

                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['total_AUC'] = roc_auc_score(
                    y_true=store_true_check, y_score=store_pred_check)

                # validation data
                # confidence
                try:
                    print(f"high_val confid {len(df_high_val_conf)}: "
                          f"{roc_auc_score(y_true=df_high_val_conf['y_true'], y_score=df_high_val_conf['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_val_AUC'] = roc_auc_score(
                        y_true=df_high_val_conf['y_true'],
                        y_score=df_high_val_conf['class_1_pred'])
                except ValueError:
                    print(f"high confid {len(df_high_val_conf)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_val_AUC'] = np.nan

                try:
                    print(f"low confid {len(df_low_val_conf)}: "
                          f"{roc_auc_score(y_true=df_low_val_conf['y_true'], y_score=df_low_val_conf['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_val_AUC'] = roc_auc_score(
                        y_true=df_low_val_conf['y_true'],
                        y_score=df_low_val_conf['class_1_pred'])
                except ValueError:
                    print(f"low confid {len(df_low_val_conf)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_val_AUC'] = np.nan

                # credibility
                try:
                    print(f"high cred {len(df_high_val_cred)}: "
                          f"{roc_auc_score(y_true=df_high_val_cred['y_true'], y_score=df_high_val_cred['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_val_cred_AUC'] = roc_auc_score(
                        y_true=df_high_val_cred['y_true'],
                        y_score=df_high_val_cred['class_1_pred'])
                except ValueError:
                    print(f"high cred {len(df_high_val_cred)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_val_cred_AUC'] = np.nan

                try:
                    print(f"low cred {len(df_low_val_cred)}: "
                          f"{roc_auc_score(y_true=df_low_val_cred['y_true'], y_score=df_low_val_cred['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_val_cred_AUC'] = roc_auc_score(
                        y_true=df_low_val_cred['y_true'],
                        y_score=df_low_val_cred['class_1_pred'])
                except ValueError:
                    print(f"low cred {len(df_low_val_cred)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_val_cred_AUC'] = np.nan

                # confidence and credibility
                try:
                    print(f"high cred {len(df_high_val_conf_cred)}: "
                          f"{roc_auc_score(y_true=df_high_val_conf_cred['y_true'], y_score=df_high_val_conf_cred['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_val_conf_cred_AUC'] = roc_auc_score(
                        y_true=df_high_val_conf_cred['y_true'],
                        y_score=df_high_val_conf_cred['class_1_pred'])
                except ValueError:
                    print(f"high cred {len(df_high_val_conf_cred)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['h_val_conf_cred_AUC'] = np.nan

                try:
                    print(f"low cred {len(df_low_val_conf_cred)}: "
                          f"{roc_auc_score(y_true=df_low_val_conf_cred['y_true'], y_score=df_low_val_conf_cred['class_1_pred'])}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_val_conf_cred_AUC'] = roc_auc_score(
                        y_true=df_low_val_conf_cred['y_true'],
                        y_score=df_low_val_conf_cred['class_1_pred'])
                except ValueError:
                    print(f"low cred {len(df_low_val_conf_cred)}: "
                          f"{np.nan}")
                    dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['l_val_conf_cred_AUC'] = np.nan

                store_true_check = df_high_val_conf['y_true'].tolist()
                store_true_check.extend(df_low_val_conf['y_true'].tolist())
                store_pred_check = df_high_val_conf['class_1_pred'].tolist()
                store_pred_check.extend(df_low_val_conf['class_1_pred'].tolist())
                print(f"total performance {len(store_pred_check)}: "
                      f"{roc_auc_score(y_true=store_true_check, y_score=store_pred_check)}")

                dict_['nonconf'][f"model_{j}"][f"s_{signif}_c_{cut_o}"]['total_val_AUC'] = roc_auc_score(
                    y_true=store_true_check, y_score=store_pred_check)

                print('\n')
                print('*' * 80)
                print('')




    print('')

    print('\n\nARGUMENTS USED')
    print(args)
    print('\n\n\n**********************************************************\n\n')

# assert 1==0, 'force stop'
import pickle

if not os.path.exists(r'/home/s185479/Python/Working_Projects/Lung_Cancer_Classification/Data/conformal_prediction_outcomes/json_files/'):
    os.makedirs(r'/home/s185479/Python/Working_Projects/Lung_Cancer_Classification/Data/conformal_prediction_outcomes/json_files/')
else:
    pass

with open(
        '/home/s185479/Python/Working_Projects/Lung_Cancer_Classification/Data/conformal_prediction_outcomes/json_files/conform_dict_lung_20220613.json',
        'wb') as fp:
    pickle.dump(dict_, fp)

if personal_on:
    with open(
            '/home/s185479/Python/Working_Projects/Lung_Cancer_Classification/Data/conformal_prediction_outcomes/json_files/personal_conform_dict_05092022_cutoff.json',
            'wb') as fp:
        pickle.dump(dict_personal, fp)
