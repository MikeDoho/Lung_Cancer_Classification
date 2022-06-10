# Python Modules
import os
import glob
import time
import numpy as np
import nrrd
import pandas as pd
import random
# Torch Modules
import torch
from torch.utils.data import Dataset
# Personal Modules
import lib.augment3D as augment3D
from lib.utils.image_manipulation import pad_to_shape, crop_center
from lib.utils.logger import log
from lib.utils.evaluation_metrics import cbct_ctsim_dose_image_review, multi_reg_image_review


# TODO: need to update from LN to primary.

class Residual_Disease_Primary(Dataset):
    """
    Base Code obtained from infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, mode,
                 dataset_path='./datasets_main',
                 exclude_mrns=[],
                 clinic_image_eval=False, tta=False):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        """
        self.mode = mode
        self.root = dataset_path

        # self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.list = []
        # self.samples = samples
        self.full_volume = None
        self.exclude_mrns = exclude_mrns
        self.args = args
        self.clinic_image_eval = clinic_image_eval
        self.tta = tta

        self.transform_tta = augment3D.RandomChoice(
            transforms=[augment3D.GaussianNoise(mean=0, std=0.01 * self.args.increase_tta_factor),
                        augment3D.RandomShift_tta(max_percentage=0.15 * self.args.increase_tta_factor),
                        augment3D.RandomRotation(min_angle=-15 * self.args.increase_tta_factor,
                                                 max_angle=15 * self.args.increase_tta_factor)],
            p=1.0)

        if self.augmentation.lower() == 'true' and self.mode == 'train':
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.02), augment3D.RandomShift(),
                            augment3D.RandomRotation(min_angle=-25, max_angle=25)], p=self.args.aug_percent)

        ### Added to gather data for clinic model training and validation###
        if self.mode == 'train':
            # Right now will not need. will save csv
            train_mrn = list(set([x.split('_')[0] for x in self.args.train_primary_list]))
            df_train = pd.DataFrame()
            df_train['mrn'] = list(set(train_mrn))

            # save file as csv
            save_location = r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/RN_pretrain_clinic/'
            train_mrn_str = f"train_mrn_primary_{self.args.random_value_for_fold[self.args.cv_count_csv_save]}.csv"
            df_train.to_csv(os.path.join(save_location, train_mrn_str), index=False)

            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            image = self.args.train_primary_list
            image = image * 2 # repeating the samples present
            labels = [int(float(x.split('_')[-1].split('-')[-1])) for x in self.args.train_primary_list]
            labels = labels * 2 # repeating the samples present

            # consider increasing the number of training examples (ie just repeat same examples)

            print('Length of training inputs: ', len(self.args.train_primary_list))
            print('Length of training image labels: ', len(labels))
            assert len(image) == len(labels), 'input and labels should be of same length'

        if self.mode == 'val':
            # Right now will not need. will save csv
            val_mrn = list(set([x.split('_')[0] for x in self.args.val_primary_list]))
            df_val = pd.DataFrame()
            df_val['mrn'] = val_mrn

            # save file as csv
            save_location = r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/RN_pretrain_clinic/'
            val_mrn_str = f"val_mrn_primary_{self.args.random_value_for_fold[self.args.cv_count_csv_save]}.csv"
            df_val.to_csv(os.path.join(save_location, val_mrn_str), index=False)

            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            image = self.args.val_primary_list
            labels = [int(float(x.split('_')[-1].split('-')[-1])) for x in self.args.val_primary_list]

            print('Length of sorted validation images: ', len(image))
            print('Length of sorted validation image labels: ', len(labels))
            assert len(image) == len(labels), 'input and labels should be of same length'

        if self.mode == 'test':
            test_mrn = list(set([x.split('_')[0] for x in self.args.test_primary_list]))
            df_test = pd.DataFrame()
            df_test['mrn'] = test_mrn

            # save file as csv
            save_location = r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/RN_pretrain_clinic/'
            test_mrn_str = f"test_mrn_primary_{self.args.random_value_for_fold[self.args.cv_count_csv_save]}.csv"
            df_test.to_csv(os.path.join(save_location, test_mrn_str), index=False)

            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            image = self.args.test_primary_list
            labels = [int(float(x.split('_')[-1].split('-')[-1])) for x in self.args.test_primary_list]

            print('Length of sorted test images: ', len(image))
            print('Length of sorted test image labels: ', len(labels))
            assert len(image) == len(labels), 'input and labels should be of same length'

        ### Added to gather data for clinic model training and validation###

        if self.mode == 'train':
            print('\n\tTraining data size: ', len(image))
            print('\n\tTraining label size: ', len(labels))
            self.list = []
            for i in range(len(image)):
                sub_list = []
                sub_list.append(image[i])
                sub_list.append(labels[i])
                self.list.append(tuple(sub_list))

        elif self.mode == 'val':
            print('\n\tValidation data size: ', len(image))
            print('\n\tValidation label size: ', len(labels))
            self.list = []
            for i in range(len(image)):
                sub_list = []
                sub_list.append(image[i])
                sub_list.append(labels[i])
                self.list.append(tuple(sub_list))

        elif self.mode == 'test':
            print('\n\tTest data size: ', len(image))
            print('\n\tTest label size: ', len(labels))
            self.list = []
            for i in range(len(image)):
                sub_list = []
                sub_list.append(image[i])
                sub_list.append(labels[i])
                self.list.append(tuple(sub_list))
                # print(self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):

        # general loading of images and label
        # TODO: no clipping has been performed

        f_img, f_label = self.list[index]
        assert int(float(f_img.split('_')[-1].split('-')[-1])) == f_label, 'check that labels match'

        # setting up loading input image; assigning label value
        # self.args.images_to_load = ['pet_ct_[', 'pet_[', 'bed']
        img, lab = np.zeros(
            (self.args.input_H, self.args.input_W, self.args.input_D, len(self.args.images_to_load))), f_label

        # print(self.args.images_to_load)
        for i, img_load_name in enumerate(self.args.images_to_load):
            # print(i, img_load_name)
            mrn = f_img.split('_')[0]
            mrn_data_dir = os.path.join(self.root, mrn)
            select_image_name = [x for x in os.listdir(mrn_data_dir) if img_load_name in x]
            if img_load_name == 'pet_[':
                select_image_name = [x for x in select_image_name if 'prert' not in x.lower()]


            # print(img_load_name)
            # print(select_image_name)
            assert len(select_image_name) == 1, f"only one file should be selected; {len(select_image_name)}"
            single_select_image_name = select_image_name[0]
            single_select_image_dir = os.path.join(mrn_data_dir, single_select_image_name)

            # loading images
            temp_img, _ = nrrd.read(single_select_image_dir)
            # nrrd.read loads images in x, y, z; going to convert to y, x, z
            temp_img = np.transpose(temp_img, [1, 0, 2])

            # adding clipping for ct for now
            if 'ct' in img_load_name:
                temp_img = np.clip(a=temp_img, a_min=self.args.ct_clip[0], a_max=self.args.ct_clip[1])
            else:
                pass

            # obtaining bounding box for selected primary disease
            # current set up hard codes for only one volume per patient
            primary_name = 'primary-1'
            # for now only using the large bounding box coordinates
            p_id = f_img.split('_')[0]

            if img_load_name in ['prert_ct_[', 'prert_pet_[']:
                temp_img = temp_img[
                           int(self.args.data_key_pre[p_id]['coord'][primary_name]['y'][0]):
                           int(self.args.data_key_pre[p_id]['coord'][primary_name]['y'][1]),
                           int(self.args.data_key_pre[p_id]['coord'][primary_name]['x'][0]):
                           int(self.args.data_key_pre[p_id]['coord'][primary_name]['x'][1]),
                           int(self.args.data_key_pre[p_id]['coord'][primary_name]['z'][0]):
                           int(self.args.data_key_pre[p_id]['coord'][primary_name]['z'][1])]


            else:
                temp_img = temp_img[
                           int(self.args.data_key_post[p_id]['coord'][primary_name]['y'][0]):
                           int(self.args.data_key_post[p_id]['coord'][primary_name]['y'][1]),
                           int(self.args.data_key_post[p_id]['coord'][primary_name]['x'][0]):
                           int(self.args.data_key_post[p_id]['coord'][primary_name]['x'][1]),
                           int(self.args.data_key_post[p_id]['coord'][primary_name]['z'][0]):
                           int(self.args.data_key_post[p_id]['coord'][primary_name]['z'][1])]

            if img_load_name in ['pet_[', 'prert_pet_[']:

                temp_img = np.clip(a=temp_img, a_min=0, a_max=self.args.image_min_max[img_load_name.replace('[', 'reshape')][1])

            # adding lazying padding and cropping but should work
            temp_img = pad_to_shape(temp_img, shape=(120, 120, 120))

            # crop to make smaller (input y, x, z)
            temp_img = crop_center(temp_img, self.args.input_H, self.args.input_W, self.args.input_D)

            img[..., i] = temp_img

        if not os.path.exists(os.path.join(r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/initial',
                                                   self.args.short_note)):
            os.makedirs(os.path.join(r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/initial',
                                                   self.args.short_note))
        else:
            pass

        if random.randint(0, 1000) > 0 and \
                mrn not in os.listdir(os.path.join(r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/initial',
                                                   self.args.short_note)):
            multi_reg_image_review(img[..., 0], img[..., 1], img[..., 2],
                                   gaps=5, initial=5, image_rows=5,
                                   mrn=mrn + '_' + self.mode + '_initial', view=False,
                                   fig_storage_dir=os.path.join(r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/initial',
                                                   self.args.short_note))

        assert np.shape(img) == (
        self.args.input_H, self.args.input_W, self.args.input_D, len(self.args.images_to_load)), \
            f" check image shape {np.shape(img)}"

        if self.args.do_normalization.lower() == 'true':

            # Will use min and max of the primary patches for now
            for i, img_load_name in enumerate(self.args.reshape_image_names):

                # Conditioning normalization parameters
                if img_load_name != 'bed_reshape' and img_load_name != 'pet_reshape' and img_load_name != 'prert_pet_reshape':
                    min_ = self.args.image_min_max[img_load_name][0]
                else:
                    # doing this because some images above are padded so the min is artificially lower
                    min_ = 0
                max_ = self.args.image_min_max[img_load_name][1]

                # this step is to address clipping; min and max were obtained prior to clipping outside of server
                if 'ct' in img_load_name:

                    if self.args.ct_clip[0] > min_:
                        min_ = self.args.ct_clip[0]
                    else:
                        pass
                    if self.args.ct_clip[1] < max_:
                        max_ = self.args.ct_clip[1]
                    else:
                        pass

                else:
                    pass

                # print(img_load_name)
                # print('global vs local min: ', min_, np.min(img[..., i]))
                # print('global vs local max: ', max_, np.max(img[..., i]))

                if self.mode == 'train':
                    assert min_ <= np.min(
                        img[..., i]), f"{img_load_name} global min {min_} vs local min {np.min(img[..., i])}, {f_img.split('_')[0]}"
                    assert max_ >= np.max(
                        img[..., i]), f"{img_load_name} global max {max_} vs local max {np.max(img[..., i])}, {f_img.split('_')[0]}"

                img[..., i] = (img[..., i] - min_) / (max_ - min_)

            if not os.path.exists(os.path.join(
                    r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/norm',
                    self.args.short_note)):
                os.makedirs(os.path.join(
                    r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/norm',
                    self.args.short_note))
            else:
                pass

            if random.randint(0, 1000) > 0 and \
                mrn not in os.listdir(os.path.join(r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/norm',
                                                   self.args.short_note)):
                multi_reg_image_review(img[..., 0], img[..., 1], img[..., 2],
                                       gaps=5, initial=5, image_rows=5,
                                       mrn=mrn + '_' + self.mode + '_norm', view=False,
                                       fig_storage_dir=os.path.join(r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/norm',
                                                   self.args.short_note))
        else:
            pass

        if not self.clinic_image_eval:
            if self.mode == 'train' and self.augmentation.lower() == 'true':
                img, lab = self.transform(img, lab)

                if not os.path.exists(os.path.join(
                        r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/aug',
                        self.args.short_note)):
                    os.makedirs(os.path.join(
                        r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/aug',
                        self.args.short_note))
                else:
                    pass

                if random.randint(0, 1000) > 0 and \
                mrn not in os.listdir(os.path.join(r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/aug',
                                                   self.args.short_note)):
                    multi_reg_image_review(img[..., 0], img[..., 1], img[..., 3],
                                           gaps=5, initial=5, image_rows=5,
                                           mrn=mrn + '_' + self.mode + '_aug', view=False,
                                           fig_storage_dir=os.path.join(r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/aug',
                                                   self.args.short_note))

            else:
                pass

            # cbct_ctsim_dose_image_review(img[..., 0], img[..., 1], img[..., 2],
            #                              save_fig_name=self.mode+'_norm_aug', view=False,
            #                              fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/')

            img = torch.FloatTensor(img)
            # Need to confirm w/ Kai but I believe this adjust the x,y,z, channel axis
            img = img.permute(3, 2, 0, 1)
            # print(torch.size(img))

            return img, lab
        else:
            if self.mode == 'train' and self.augmentation.lower() == 'true':
                img, lab = self.transform(img, lab)

                if random.randint(0, 1000) > 1001:
                    multi_reg_image_review(img[..., 0], img[..., 1], img[..., 2],
                                           gaps=5, initial=5, image_rows=5,
                                           mrn=mrn + '_' + self.mode + '_aug2', view=False,
                                           fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/primary_site_eval/aug')

            elif self.tta:

                # performing all three transformations
                transforms_selection = [augment3D.GaussianNoise(mean=0, std=0.01 * self.args.increase_tta_factor),
                                        augment3D.RandomShift_tta(max_percentage=0.15 * self.args.increase_tta_factor),
                                        augment3D.RandomRotation(min_angle=-15 * self.args.increase_tta_factor,
                                                                 max_angle=15 * self.args.increase_tta_factor)]

                # there is a random.sample function in ComposeTransforms_tta
                # will select 1, 2, or all 3 transformations and apply the transformations selected
                all_transform = augment3D.ComposeTransforms_tta(transforms=transforms_selection)
                img, lab = all_transform(img, lab)

            else:
                pass

            # cbct_ctsim_dose_image_review(img[..., 0], img[..., 1], img[..., 2],
            #                              save_fig_name=self.mode, view=False,
            #                              fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/')

            img = torch.FloatTensor(img)
            # Need to confirm w/ Kai but I believe this adjusts the x,y,z, channel axis
            img = img.permute(3, 2, 0, 1)

            return img, lab, f_img.split('_')[0]
