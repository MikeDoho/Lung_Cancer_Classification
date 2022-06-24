#!/bin/bash

python Train_Transfer_Lung_CV_evdl.py >> Logs/20220621_res101ext-do_ds020bn005-evdl_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_ep[150]_cw[1.0,1.0]_sw[1.0,1.0]_s[200]_c[na]_[none]_im[ct]_cv[5].log \
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
--short_note "20220621_101ext-do_ds020bn005-evdl_lTL[1e-2]_lr[wcos30_1e-2]_a[0.7]_e[150]_s[200]_cv[5]" \
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
--aug_percent 0.70

