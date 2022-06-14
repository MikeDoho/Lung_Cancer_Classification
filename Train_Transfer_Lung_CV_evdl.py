# Python libraries
import os
import datetime
import time
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys

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
# import lib.Trainers.pytorch_trainer_pretrain_clinic_softmax_lung as pytorch_trainer
import lib.Trainers.pytorch_trainer_evdl_lung as pytorch_trainer
from lib.utils.logger import log
from lib.medzoo.ResNet3DMedNet import generate_resnet3d
from lib.Models.resnet_fork_single_input_extend_dropout import resnet50

# try to address speed issues?
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = True

'''

Current adjustments to the code from baseline: using model_spatial instead of model in the module section.
Will have to revert. Will also have to confirm new transfer learning layer selections is appropriate. 

if change back to prior method then will have to change # --new_layer_names avgpool dropout fc 

'''



def main():
    time_stamp = datetime.datetime.now()
    print("Time stamp " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'), '\n\n')

    print("Arguments Used")
    args = parse_opts()
    args.time_stamp = time_stamp.strftime('%Y.%m.%d')

    # import sys
    # f = open(f"/home/s185479/Python/Working_Projects/Lung_Cancer_Classification/Logs/{args.time_stamp}_{args.short_note}.log", 'w')

    # logging.basicConfig(
    #     filename=f"/home/s185479/Python/Working_Projects/Lung_Cancer_Classification/Logs/{args.time_stamp}_{args.short_note}.log",
    #     format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S',
    #     level=logging.DEBUG)
    #
    # log = logging.getLogger()


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

            print('Creating validation sampler for less errors displayed in log file; more permanent solution needed')
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

        # TRAINING
        training_dataset = Lung_Cancer_Classification(args,
                                                      mode='train',
                                                      dataset_path=args.main_input_dir,
                                                      exclude_mrns=exclude_mrns)
        training_generator = DataLoader(training_dataset,
                                        batch_size=args.batch_size,
                                        sampler=sampler,
                                        shuffle=shuffle_,  # shuffle cannot be on with a sampler
                                        pin_memory=args.pin_memory,
                                        num_workers=args.num_workers)

        # VALIDATION
        validation_dataset = Lung_Cancer_Classification(args,
                                                        mode='val',
                                                        dataset_path=args.main_input_dir,
                                                        exclude_mrns=exclude_mrns)
        val_generator = DataLoader(validation_dataset,
                                   batch_size=args.batch_size,
                                   sampler=sampler_val,
                                   shuffle=shuffle_,  # shuffle cannot be on with a sampler
                                   num_workers=0)

        # TESTING
        test_dataset = Lung_Cancer_Classification(args,
                                                  mode='test',
                                                  dataset_path=args.main_input_dir,
                                                  exclude_mrns=exclude_mrns)
        test_generator = DataLoader(test_dataset,
                                    batch_size=1,
                                    sampler=None,
                                    shuffle=True,
                                    num_workers=0)

        # Setting model and optimizer
        print('')
        # torch.manual_seed(sets.manual_seed) # already set above

        if 'basicresnet50' not in args.short_note:
            model, parameters = generate_model(args)
            summary(model, (args.in_modality, args.input_D, args.input_H, args.input_W))

            if args.ci_test:
                params = [{'params': parameters, 'lr': args.learning_rate}]
            else:
                params = [
                    {'params': parameters['base_parameters'], 'lr': args.learning_rate * args.resnet_lr_factor},
                    {'params': parameters['new_parameters'], 'lr': args.learning_rate}
                ]

            print('rn lr: ', float(args.resnet_lr_factor))
            print('exclude type: ', type(args.exclude_mrn))

            for k, v in model.named_parameters():
                temp = k[7:]
                weight = temp[:-7]
                bias = temp[:-5]
                if weight in args.new_layer_names:
                    # print('{} skipped'.format(k))
                    continue
                elif bias in args.new_layer_names:
                    # print('{} skipped'.format(k))
                    continue

                else:
                    # print('RequireGrad for {} set False'.format(k))
                    if float(args.resnet_lr_factor) == 0.0:
                        v.requires_grad = False
                    else:
                        # v.requires_grad = True
                        pass

            optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)

        else:
            print('Creating basic resnet50')
            # model = generate_resnet3d()
            model = resnet50(sample_input_W=args.input_W,
                             sample_input_H=args.input_H,
                             sample_input_D=args.input_D,
                             shortcut_type=args.resnet_shortcut,
                             no_cuda=args.no_cuda,
                             num_classes=args.n_classes)
            summary(model, (args.in_modality, args.input_D, args.input_H, args.input_W))
            params = model.parameters()
            optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6,
                                                                         last_epoch=-1)
        # print('parameters (look here): \n', params)

        if args.resume_path:
            if os.path.isfile(args.resume_path):

                if not os.path.exists(args.resume_path):
                    os.makedirs(args.resume_path)
                else:
                    pass

                print("=> loading checkpoint '{}'".format(arg.resume_path))
                checkpoint = torch.load(arg.resume_path)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(arg.resume_path, checkpoint['epoch']))
        # print(f"\t# Trainable model parameters: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")

        # Selecting loss function
        criterion_pre = nn.CrossEntropyLoss(ignore_index=-1,
                                            weight=torch.FloatTensor(ast.literal_eval(args.class_weights)))

        criterion_pre = criterion_pre.to(torch.device('cuda'))

        # tb_logger = SummaryWriter('./events/{}'.format(args.short_note))
        if args.use_tb.lower() == 'true':
            tb_logger = SummaryWriter(f"./pretrain_clinic/{args.short_note}_{args.resnet_lr_factor}_" \
                                      f"{args.input_H, args.input_W, args.input_D}_exclude_{str(args.exclude_mrn)}_" \
                                      f"{args.exclude_mrn_filename.split('_')[-2]}_{args.class_weights}-{args.time_stamp}-{args.true_cv_count + 1}_of_{args.cv_num}_"
                                      f"seed_{args.manual_seed}")
        else:
            tb_logger = None

        args.true_cv_count += 1

        print("Assessing GPU usage")
        if args.cuda:
            print(f"\tCuda set to {args.cuda}\n")
            model = model.to(torch.device('cuda'))
        # summary(model, (args.in_modality, args.input_D, args.input_H, args.input_W))

        print("Initializing training")
        trainer = pytorch_trainer.Trainer(args, model, criterion_pre, optimizer, train_data_loader=training_generator, \
                                          valid_data_loader=val_generator, test_data_loader=test_generator,
                                          lr_scheduler=scheduler, tb_logger=tb_logger)

        print("Start training!")
        trainer.training()

        args.cv_count_csv_save += 1

    # sys.stdout.close()

if __name__ == '__main__':
    main()
