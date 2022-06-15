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
import lib.Trainers.pytorch_trainer_evdl_lung as pytorch_trainer
from lib.utils.logger import log
from lib.medzoo.ResNet3DMedNet import generate_resnet3d
from lib.Models.resnet_fork_single_input_extend_dropout import resnet50

# try to address speed issues?
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = True

time_stamp = datetime.datetime.now()
print("Time stamp " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'), '\n\n')

print("Arguments Used")
args = parse_opts()
args.time_stamp = time_stamp.strftime('%Y.%m.%d')

# if float(args.resnet_lr_factor) > 0:
#     print('\ndownsizing batch size because now we are training the whole model\n')
#     args.batch_size = 16
#     print(args.batch_size)
# else:
#     pass

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

# model_to_load = '20220610_res101ext-sm-do005_lTL[1e-2]_lr[wcos30_1e-2]_a[0.9]_ep[100]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV'
model_to_load = '20220614_res101ext-sm-do005_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[200]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[3]_CV'

checkpoint_path_main = '/home/s185479/Python/Working_Projects/Lung_Cancer_Classification/trails/' + \
                       model_to_load

# saved_end_model_120 = [
#     'resnet_101_epoch_89_20220610_res101ext-sm-do005_lTL[1e-2]_lr[wcos30_1e-2]_a[0.9]_ep[100]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV_fold_1.pth.tar',
#     'resnet_101_epoch_98_20220610_res101ext-sm-do005_lTL[1e-2]_lr[wcos30_1e-2]_a[0.9]_ep[100]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV_fold_2.pth.tar',
#     'resnet_101_epoch_84_20220610_res101ext-sm-do005_lTL[1e-2]_lr[wcos30_1e-2]_a[0.9]_ep[100]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV_fold_3.pth.tar',
#     'resnet_101_epoch_98_20220610_res101ext-sm-do005_lTL[1e-2]_lr[wcos30_1e-2]_a[0.9]_ep[100]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV_fold_4.pth.tar',
#     'resnet_101_epoch_90_20220610_res101ext-sm-do005_lTL[1e-2]_lr[wcos30_1e-2]_a[0.9]_ep[100]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[5]_CV_fold_5.pth.tar']

saved_end_model_120 = [
    'resnet_101_epoch_143_20220614_res101ext-sm-do005_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[200]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[3]_CV_fold_1.pth.tar',
    'resnet_101_epoch_179_20220614_res101ext-sm-do005_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[200]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[3]_CV_fold_2.pth.tar',
    'resnet_101_epoch_191_20220614_res101ext-sm-do005_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[200]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[wsamp]_im[ct]_cv[3]_CV_fold_3.pth.tar'
    ]

print('saved model: ', saved_end_model_120)

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


    def set_dropout(model, drop_rate=0.1, reset_chance=0.50):
        for name, child in model.named_children():
            if isinstance(child, torch.nn.Dropout3d):
                x = np.random.random()
                print(child, ' old DO rate: ', child.p, x, x<=reset_chance)
                if x <= reset_chance:
                    child.p = drop_rate
                else:
                    child.p = 0.0
                print(child, ' new DO rate: ', child.p)
            set_dropout(child, drop_rate=drop_rate, reset_chance=reset_chance)


    model_do_copy = copy.deepcopy(model)

    # # setting new dropout value
    # print('Changing dropout for Epistemic Uncertainty')
    # print(f"reset dropout percent: {args.reset_dropout_percent}")
    # print(f"mc_dropout_percent: {args.mc_dropout_percent}")
    # set_dropout(model_do_copy, drop_rate=args.mc_dropout_percent, reset_chance=args.reset_dropout_percent)
    # print('')

    # This is a separate function to tune the dropout in the bottleneck and the in sequential layers
    def specialized_model_do_rate_set(model_do_copy,
                                      pp_bottleneck_rate=0.45, bottleneck_reset=1,
                                      pp_downsample_rate=0.33, downsample_reset=1):

        bottleneck_reset = np.clip(bottleneck_reset, 0, 1)
        pp_bottleneck_rate = np.clip(pp_bottleneck_rate, 0, 1)

        downsample_reset = np.clip(downsample_reset, 0, 1)
        pp_downsample_rate = np.clip(pp_downsample_rate, 0, 1)


        for m in model_do_copy.modules():
            # if m.__class__.__name__ == 'Bottleneck':
            #     if np.random.random() <= bottleneck_reset:
            #         m.dropout3d_bottleneck.p = pp_bottleneck_rate
            #     else:
            #         m.dropout3d_bottleneck.p = 0
            if m.__class__.__name__ == 'Sequential':
                if np.random.random() <= downsample_reset:
                    m[1].p = pp_downsample_rate
                else:
                    m[1].p = 0
            else:
                pass


    specialized_model_do_rate_set(model_do_copy=model_do_copy,
                                  pp_bottleneck_rate=args.mc_bottleneck_dropout_rate, bottleneck_reset=args.reset_bottleneck_dropout_percent,
                                  pp_downsample_rate=args.mc_downsample_dropout_rate, downsample_reset=args.reset_downsample_dropout_percent)


    print_model = True
    if print_model:
        print('\n*********************\nNeed to evaluate the model')
        for name, child in model_do_copy.named_children():
            print(name, child)

        # print('\nNamed modules')
        # for idx, m in enumerate(model_do_copy.named_modules()):
        #     print(idx, '->', m)

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
        args.train_mrn_list = [x for x in xtrain_ if x not in exclude_mrns]
        args.val_mrn_list = [x for x in xval_ if x not in exclude_mrns]
        args.test_mrn_list = [x for x in xtest_ if x not in exclude_mrns]

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
    validation_dataset_eval = Lung_Cancer_Classification(args,
                                                    mode='val',
                                                    dataset_path=args.main_input_dir,
                                                    exclude_mrns=exclude_mrns,
                                                    clinic_image_eval=True)
    val_generator_eval = DataLoader(validation_dataset_eval,
                               batch_size=1,
                               sampler=None,
                               shuffle=True,  # shuffle cannot be on with a sampler
                               num_workers=0)

    # For TTA
    validation_dataset_eval_tta = Lung_Cancer_Classification(args,
                                                    mode='val',
                                                    dataset_path=args.main_input_dir,
                                                    exclude_mrns=exclude_mrns,
                                                    clinic_image_eval=True, tta=True)
    val_generator_eval_tta = DataLoader(validation_dataset_eval_tta,
                               batch_size=1,
                               sampler=None,
                               shuffle=True,  # shuffle cannot be on with a sampler
                               num_workers=0)

    # TESTING
    test_dataset_eval = Lung_Cancer_Classification(args,
                                                    mode='test',
                                                    dataset_path=args.main_input_dir,
                                                    exclude_mrns=exclude_mrns,
                                                    clinic_image_eval=True)
    test_generator_eval = DataLoader(test_dataset_eval,
                               batch_size=1,
                               sampler=None,
                               shuffle=True,  # shuffle cannot be on with a sampler
                               num_workers=0)

    # For TTA
    test_dataset_eval_tta = Lung_Cancer_Classification(args,
                                                    mode='test',
                                                    dataset_path=args.main_input_dir,
                                                    exclude_mrns=exclude_mrns,
                                                    clinic_image_eval=True, tta=True)

    test_generator_eval_tta = DataLoader(test_dataset_eval_tta,
                               batch_size=1,
                               sampler=None,
                               shuffle=True,  # shuffle cannot be on with a sampler
                               num_workers=0)

    if args.mc_do.lower() == 'true':
        # Epistemic Uncertainty # Calculating MC Dropout
        from lib.utils.mc_dropout_evdl_pytorch import get_monte_carlo_predictions

        df_val_epistemic_entropy = get_monte_carlo_predictions(data_loader=val_generator_eval,
                                                               forward_passes=args.mc_do_num,
                                                               model=model_do_copy,
                                                               n_classes=2,
                                                               n_samples=len(val_generator_eval))

        df_test_epistemic_entropy = get_monte_carlo_predictions(data_loader=test_generator_eval,
                                                                forward_passes=args.mc_do_num,
                                                                model=model_do_copy,
                                                                n_classes=2,
                                                                n_samples=len(test_generator_eval))

        epistemic_uncertainty_store_dir = './Data/Uncertainty/Epistemic/'

        if not os.path.exists(epistemic_uncertainty_store_dir):
            os.makedirs(epistemic_uncertainty_store_dir)
        else:
            pass

        # Saving validation Epistemic uncertainty calculations

        filename_val_epistemic_entropy = f"image_model_val_evdl_epistemic_do_entropy_" \
                                         f"resnet_lung_bestof_fold_{j + 1}_bottleneck_{args.reset_bottleneck_dropout_percent}_" \
                                         f"{args.mc_bottleneck_dropout_rate}_downsample_{args.reset_downsample_dropout_percent}_" \
                                         f"{args.mc_downsample_dropout_rate}_num{args.mc_do_num}_rundate_{args.time_stamp}.csv"
        val_epistemic_entropy_csv_dir = os.path.join(epistemic_uncertainty_store_dir,
                                                     filename_val_epistemic_entropy)
        df_val_epistemic_entropy.to_csv(val_epistemic_entropy_csv_dir)

        # Saving test Epistemic uncertainty calculations
        filename_test_epistemic_entropy = f"image_model_test_evdl_epistemic_do_entropy_" \
                                          f"resnet_lung_bestof_fold_{j + 1}_bottleneck_{args.reset_bottleneck_dropout_percent}_" \
                                          f"{args.mc_bottleneck_dropout_rate}_downsample_{args.reset_downsample_dropout_percent}_" \
                                          f"{args.mc_downsample_dropout_rate}_num{args.mc_do_num}_rundate_{args.time_stamp}.csv"
        test_epistemic_entropy_csv_dir = os.path.join(epistemic_uncertainty_store_dir,
                                                      filename_test_epistemic_entropy)
        df_test_epistemic_entropy.to_csv(test_epistemic_entropy_csv_dir)

    if args.do_tta.lower() == 'true':
        # Aleatoric Uncertainty
        from lib.utils.tta_evdl_pytorch import get_tta_predictions

        df_val_aleatoric_entropy = get_tta_predictions(data_loader=val_generator_eval_tta,
                                                       forward_passes=args.tta_num,
                                                       model=model,
                                                       n_classes=2,
                                                       n_samples=len(val_generator_eval_tta))

        df_test_aleatoric_entropy = get_tta_predictions(data_loader=test_generator_eval_tta,
                                                        forward_passes=args.tta_num,
                                                        model=model,
                                                        n_classes=2,
                                                        n_samples=len(test_generator_eval_tta))

        aleatoric_uncertainty_store_dir = './Data/Uncertainty/Aleatoric/'

        if not os.path.exists(aleatoric_uncertainty_store_dir):
            os.makedirs(aleatoric_uncertainty_store_dir)
        else:
            pass

        # Saving validation aleatoric uncertainty calculations
        filename_val_aleatoric_entropy = f"image_model_val_evdl_aleatoric_tta_entropy_" \
                                         f"resnet_lung_bestof_fold_{j + 1}_num{args.tta_num}_" \
                                         f"ttafactor_{args.increase_tta_factor}_rundate_{args.time_stamp}.csv"
        val_aleatoric_entropy_csv_dir = os.path.join(aleatoric_uncertainty_store_dir,
                                                     filename_val_aleatoric_entropy)
        df_val_aleatoric_entropy.to_csv(val_aleatoric_entropy_csv_dir)

        # Saving test aleatoric uncertainty calculations
        filename_test_aleatoric_entropy = f"image_model_test_evdl_aleatoric_tta_entropy_" \
                                          f"resnet_lung_bestof_fold_{j + 1}_num{args.tta_num}_" \
                                          f"ttafactor_{args.increase_tta_factor}_rundate_{args.time_stamp}.csv"
        test_aleatoric_entropy_csv_dir = os.path.join(aleatoric_uncertainty_store_dir,
                                                      filename_test_aleatoric_entropy)
        df_test_aleatoric_entropy.to_csv(test_aleatoric_entropy_csv_dir)

        # will need to add a basic model prediction component

    print('\n\nARGUMENTS USED')
    print(args)
    print('\n\n\n**********************************************************\n\n')
