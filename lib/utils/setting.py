'''
Configs for training and testing residual disease model

Will need to organize parameters
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    '''
    
    Model note
    
    '''

    parser.add_argument(
        '--short_note',
        default='',
        type=str,
        help='short note in model saving name')

    '''
    
    Data Directories
    
    '''

    parser.add_argument(
        '--data_root',
        default='./data',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--main_input_dir',
        default='./data',
        type=str,
        help='location of input data; contains pet, ct, dose, bed')
    parser.add_argument(
        '--train_dataset_path',
        default='./data',
        type=str,
        help='Path to training images')
    parser.add_argument(
        '--train_label_path',
        default='./data',
        type=str,
        help='Path to training labels')
    parser.add_argument(
        '--val_dataset_path',
        default='./data',
        type=str,
        help='Path to validation images')
    parser.add_argument(
        '--val_label_path',
        default='./data',
        type=str,
        help='Path to validation labels')
    parser.add_argument(
        '--test_dataset_path',
        default='./data',
        type=str,
        help='Path to test images')
    parser.add_argument(
        '--test_label_path',
        default='./data',
        type=str,
        help='Path to test labels')
    parser.add_argument(
        '--data_key_dir',
        default='./data_key',
        type=str,
        help='Path to data key which includes bounding box, labels, min max of inputs')
    parser.add_argument(
        '--data_key_filename',
        default='unspecified_data_key.json',
        type=str,
        help='data key filename')
    parser.add_argument(
        '--data_key_filename_pre',
        default='unspecified_data_key.json',
        type=str,
        help='data key filename prert')
    parser.add_argument(
        '--data_key_filename_post',
        default='unspecified_data_key.json',
        type=str,
        help='data key filename postrt')
    parser.add_argument(
        '--exclude_mrn',
        default='false',
        type=str,
        help='should we exclude MRNs of selected patients')
    parser.add_argument(
        '--exclude_mrn_path',
        default=r'./path',
        type=str,
        help='Path exclude mrn files')
    parser.add_argument(
        '--exclude_mrn_filename',
        default='na.csv',
        type=str,
        help='file to load which holds list of MRNs to exclude')
    parser.add_argument(
        '--fraction_key_path',
        default='./path/',
        type=str,
        help='Path fraction key')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.')
    parser.add_argument(
        '--pretrain_path',
        default='pretrain/resnet_50.pth',
        type=str,
        help=
        'Path for pretrained model.')
    parser.add_argument(
        '--min_max_key_path',
        default='./path/',
        type=str,
        help='Path min max key')

    '''
    
    Input Parameters
    
    '''

    parser.add_argument(
        '--input_D',
        default=56,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=448,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=448,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--small_radius_filter_mm',
        default=3.5,
        type=float,
        help='filter criteria to remove LN with approximate volume associated with the radius in mm selected')
    parser.add_argument(
        '--images_to_load',
        default=['pet_ct_[', 'pet_[', 'bed'],
        # type=list,
        nargs='+',
        help='images to load into model')
    parser.add_argument(
        '--in_modality',
        default=1,
        type=int,
        help="number of imaging modalities in the input")
    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help="Number of classification classes")

    '''
    
    Model parameters
    
    '''

    parser.add_argument(
        '--is_classi',
        default=False,
        type=bool,
        help="Transfer learning or not")
    parser.add_argument(
        '--is_transfer',
        default=False,
        type=bool,
        help="Transfer learning or not")
    parser.add_argument(
        '--n_seg_classes',
        default=2,
        type=int,
        help="Number of segmentation classes")
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--gamma',  # set to 0.001 when finetune
        default=0.95,
        type=float,
        help=
        'Initial gamma')
    parser.add_argument(
        '--num_workers',
        default=0,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--new_layer_names',
        #default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
        default=['conv_seg'],
        nargs='+',
        # type=list,
        help='New layer except for backbone')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument(
        '--cuda', action='store_true', help='If true, cuda is used.')
    parser.set_defaults(no_cuda=False, cuda=True)

    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, ci testing is used.')
    parser.add_argument(
        '--augmentation', default='false', help='if use augmentations during training')
    parser.add_argument(
        '--resnet_lr_factor',
        default=0,
        type=float,
        help='slightly train pretrained weights; should be less than 1')
    parser.add_argument(
        '--class_weights',
        default="[1.0, 1.0]",
        type=str,
        help='class weights for loss function; shape needs to match number of classes')
    parser.add_argument(
        '--cv_num',
        default=1,
        type=int,
        help='number of cross validation performed in the image/clinic model')
    parser.add_argument(
        '--do_normalization',
        default='false',
        type=str,
        help='number of cross validation performed in the image/clinic model')
    parser.add_argument(
        '--aug_percent',
        default=0.05,
        type=float,
        help='number of cross validation performed in the image/clinic model')
    parser.add_argument(
        '--ct_clip',
        default=[-300, 600],
        nargs='+',
        # type=float,
        help='number of cross validation performed in the image/clinic model')
    parser.add_argument(
        '--weighted_sampler_on',
        default='true',
        # nargs='+',
        type=str,
        help='use weighted sampler to address class imbalance; will attempt to make classes 50/50 in binary')
    parser.add_argument(
        '--weighted_sampler_weight_adjust',
        default='[1, 1]',
        # nargs='+',
        type=str,
        help='default setting of weighted sampler is to make 50/50; can make relative adjustments with these values')

    '''
    
    Clinical model parameters
    
    '''

    parser.add_argument(
        '--clinical_data_path',
        default='/data/maia/mdohop/Holman_Pathway/Residual_Disease/Data/rd_ln_data_1/clinical_data',
        type=str,
        help='PATH TO CLINICAL DATA')
    parser.add_argument(
        '--clinical_data_filename',
        default='processed_unnormalized_data_350.csv',
        type=str,
        help='CLINICAL DATA FILENAME')
    parser.add_argument(
        '--selected_clinical_features',
        default=['Age at Diagnosis', 'Clinical N Stage', 'Clinical T Stage', 'Concurrent Chemo Regimen_cat_4', \
                 'Ethnicity_cat_1', 'ae_prior_select_date'],
        help='LIST OF SELECTED CLINICAL FEATURES')
    parser.add_argument(
        '--clinical_model_type',
        default='lr',
        type=str,
        help='select clinical model to use (only LR and SVM at the moment)')
    parser.add_argument(
        '--clinical_model_on',
        default='false',
        type=str,
        help='whether or not to use clinical model')

    '''
    
    Tensorboard
    
    '''

    parser.add_argument(
        '--use_tb',
        default='false',
        type=str,
        help='True if want to store input into tensorboard')

    '''
    
    Uncertainty Parameters
    
    '''

    parser.add_argument(
        '--mc_do',
        default='false',
        type=str,
        help='selecting if MC dropout should be performed')
    parser.add_argument(
        '--mc_do_num',
        default=50,
        type=int,
        help='number of dropout MC performed')
    parser.add_argument(
        '--do_tta',
        default='false',
        type=str,
        help='selecting if TTA should be performed')
    parser.add_argument(
        '--tta_num',
        default=50,
        type=int,
        help='number of tta performed')
    parser.add_argument(
        '--reset_dropout_percent',
        default=0.5,
        type=float,
        help='for MC DO testing predictions; chance of reset')
    parser.add_argument(
        '--mc_dropout_percent',
        default=0.1,
        type=float,
        help='for MC DO rate testing predictions')
    parser.add_argument(
        '--reset_bottleneck_dropout_percent',
        default=1,
        type=float,
        help='for MC DO testing predictions; chance of reset bottleneck dropout')
    parser.add_argument(
        '--mc_bottleneck_dropout_rate',
        default=0.03,
        type=float,
        help='for MC DO rate testing predictions')
    parser.add_argument(
        '--reset_downsample_dropout_percent',
        default=1,
        type=float,
        help='for MC DO testing predictions; chance of reset downsample dropout')
    parser.add_argument(
        '--mc_downsample_dropout_rate',
        default=0.03,
        type=float,
        help='for MC DO rate testing predictions')
    parser.add_argument(
        '--increase_tta_factor',
        default=1,
        type=float,
        help='for MC DO rate testing predictions')


    args = parser.parse_args()
    args.save_folder = "./trails/{}/{}_{}".format(args.short_note, args.model, args.model_depth)
    
    print(args)
    return args
