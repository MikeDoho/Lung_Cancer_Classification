#!/bin/bash

python Trained_lung_evdl_model_uncertainty.py >> Logs/20220622_trained-evdl_bs20bn5_DO-100-005_100-020_do200_tta200-3_cv5.log \
--gpu_id 1 \
--is_transfer True \
--is_classi True \
--in_modality 1 \
--n_epochs 150 \
--manual_seed 200 \
--batch_size 30 \
--input_H 32 \
--input_W 32 \
--input_D 32 \
--model resnet \
--model_depth 101 \
--num_workers 0 \
--pretrain_path lib/Models/pretrain/resnet_101_23dataset.pth \
--resnet_shortcut B \
--new_layer_names conv_pred \
\
--images_to_load ct \
--main_input_dir '/home/s185479/Python/Working_Data/Lung_Cancer_Classification/Data_Label/' \
--train_dataset_path "na" \
--train_label_path "na" \
--val_dataset_path "na" \
--val_label_path "na" \
--test_dataset_path "na" \
--test_label_path "na" \
--min_max_key_path "na" \
--fraction_key_path "na" \
\
--weighted_sampler_on False \
--weighted_sampler_weight_adjust '[1.0, 1.0]' \
\
--clinical_model_on 'false' \
--clinical_data_path './' \
--clinical_data_filename 'na.csv' \
--selected_clinical_features "['Smoking Status', 'Clinical N Stage', 'Clinical T Stage','Concurrent Chemo Regimen_cat_4', 'HPV Status']" \
--clinical_model_type 'lr' \
\
--use_tb 'false' \
\
--short_note "20220622_trained-evdl_res101_bs20bn5_DO-100-005_100-020_do200_tta200-3_cv5" \
--exclude_mrn 'false' \
--exclude_mrn_filename 'na.csv' \
--exclude_mrn_path './na/' \
--resnet_lr_factor 0.01 \
\
--learning_rate 0.01 \
\
--class_weights "[1.0, 1.0]" \
--cv_num 5 \
--augmentation 'true' \
--do_normalization 'true' \
--aug_percent 0.70 \
\
--reset_bottleneck_dropout_percent 1 \
--mc_bottleneck_dropout_rate 0.05 \
--reset_downsample_dropout_percent 1 \
--mc_downsample_dropout_rate 0.20 \
--mc_do 'true' \
--do_tta 'true' \
--mc_do_num 300 \
--tta_num 300 \
--increase_tta_factor 3 \
