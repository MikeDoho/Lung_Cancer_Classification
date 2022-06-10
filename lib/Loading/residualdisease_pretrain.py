# Python Modules
import os
import glob
import time
import numpy as np
import pandas as pd
# Torch Modules
import torch
from torch.utils.data import Dataset
# Personal Modules
import lib.augment3D as augment3D
from lib.utils.logger import log
from lib.utils.evaluation_metrics import cbct_ctsim_dose_image_review
from lib.utils.data_process import dilate_mask


class RESIDUALDISEASE(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, mode, dataset_path='./datasets_main', label_path='./datasets', classes=2,
                 exclude_mrns=[], train_path='./datasets_train', val_path='./datasets_val', test_path='./datasets_test',
                 clinic_image_eval=False, aug_percent=0.90):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = dataset_path
        self.label_path = label_path
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.list = []
        # self.samples = samples
        self.full_volume = None
        self.classes = classes
        self.exclude_mrns = exclude_mrns
        self.args = args
        self.clinic_image_eval = clinic_image_eval
        self.aug_percent = aug_percent

        # self.train_mrn_list = train_mrn_list
        # self.val_mrn_list = val_mrn_list
        # self.test_mrn_list = test_mrn_list

        if self.augmentation.lower() == 'true' and self.mode == 'train':
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomShift(),
                            augment3D.RandomRotation(min_angle=-25, max_angle=25)], p=self.aug_percent)

        # print(self.root)
        # image = glob.glob(os.path.join(self.root, '*.npy'))
        train_image = glob.glob(os.path.join(self.train_path, '*.npy'))
        val_image = glob.glob(os.path.join(self.val_path, '*.npy'))
        test_image = glob.glob(os.path.join(self.test_path, '*.npy'))
        # print(self.train_path)
        # print(train_image)

        image = train_image + val_image + test_image
        print('len of all image directories: ', len(image))
        # print(image)

        # print(f"\tLoading {self.mode} data from", os.path.join(self.root, '*.npy'))

        image = [x for x in image if x.split('/')[-1].split('.')[0].split('_')[0] not in self.exclude_mrns]

        print('len of all image directories after excluding mrns: ', len(image))

        labels = []
        for path in image:
            temp = path.split('/')[-1]
            temp = temp.split('.')[0]
            temp = temp.split('_')[0]
            labels += glob.glob(os.path.join(self.label_path, temp + '_outcome.npy'))

        # print(glob.glob(os.path.join(self.label_path, '*.npy')))
        # print(labels)

        labels = [x for x in labels if x.split('/')[-1].split('.')[0].split('_')[0] not in self.exclude_mrns]

        ### Added to gather data for clinic model training and validation###
        if self.mode == 'train':
            # load outcome key if want to do oversampling; creating list to increase number of instances
            outcome_key_df = \
                pd.read_csv(
                    r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Data/rd_data_1/outcome_key/Outcome_key.csv').iloc[:, 1:]
            oversample_list = outcome_key_df.loc[
                outcome_key_df['Results of Post-Radiation Neck Dissection'] == 1, 'MRN'].tolist()

            oversample_list = list(map(str, oversample_list))

            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            image = [x for x in image if x.split('/')[-1].split('.')[0].split('_')[0] in self.args.train_mrn_list and \
                     not len(x.split('/')[-1].split('.')[0].split('_')) > 1]
            image = sorted(image, key=lambda x: x.split('/')[-1].split('.')[0].split('_')[0])

            labels = [x for x in labels if x.split('/')[-1].split('.')[0].split('_')[0] in self.args.train_mrn_list]
            labels = sorted(list(set(labels)), key=lambda x: x.split('/')[-1].split('.')[0].split('_')[0])

            image_copy = image.copy()
            labels_copy = labels.copy()

            # print('oversample value: ', self.args.oversample)
            # print(image[0:3])

            ###OVERSAMPLING###
            if self.args.oversample > 0:
                # print('oversample is greater than 0')
                print(oversample_list)
                for train in (train for train in image if train.split('/')[-1].split('.')[0].split('_')[0] in oversample_list):
                    # print('oversample list is recognized')
                    add = []
                    add_labels = []


                    # adds oversample_value additions of train (train only includes residual disease + patients)
                    for i in range(self.args.oversample):
                        add.append(train)
                        add_labels.append(os.path.join(self.label_path,
                                                       train.split('/')[-1].split('.')[0].split('_')[0]+'_outcome.npy'))


                    image_copy = image_copy + add
                    # image_copy.extend(add)
                    # labels_copy.extend(add_labels)
                    labels_copy = labels_copy + add_labels

                # filter_train = train_x_choices_copy
                image = image_copy
                print(f"length of oversample presort: {len(image)}")
                image = sorted(image, key=lambda x: x.split('/')[-1].split('.')[0].split('_')[0])
                labels = labels_copy
                print(f"length of oversample pre sort: {len(labels)}")
                labels = sorted(labels, key=lambda x: x.split('/')[-1].split('.')[0].split('_')[0])

                print(f"length of oversample: {len(image)}")
                print(f"length of oversample: {len(labels)}")

            else:
                # filter_train = train_x_choices
                pass
            ###OVERSAMPLING###

            print('Length of training images: ', len(image))
            log.info('train')
            log.info(len(image))
            log.info(len(labels))
            print('Length of training image labels: ', len(labels))

        if self.mode == 'val':
            print('len val old: ', np.shape(image))
            image = [x for x in image if x.split('/')[-1].split('.')[0].split('_')[0] in self.args.val_mrn_list and \
                     not len(x.split('/')[-1].split('.')[0].split('_')) > 1]
            image = sorted(image, key=lambda x: x.split('/')[-1].split('.')[0].split('_')[0])

            labels = [x for x in labels if x.split('/')[-1].split('.')[0].split('_')[0] in self.args.val_mrn_list]
            labels = sorted(list(set(labels)), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0]))

            print('Length of sorted validation images: ', len(image))
            print('Length of sorted validation image labels: ', len(labels))

        if self.mode == 'test':
            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            print('Length of old test images: ', len(image))
            image = [x for x in image if x.split('/')[-1].split('.')[0].split('_')[0] in self.args.test_mrn_list and \
                     not len(x.split('/')[-1].split('.')[0].split('_')) > 1]
            image = sorted(image, key=lambda x: x.split('/')[-1].split('.')[0].split('_')[0])

            labels = [x for x in labels if x.split('/')[-1].split('.')[0].split('_')[0] in self.args.test_mrn_list]
            labels = sorted(list(set(labels)), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0]))

            print('Length of sorted test images: ', len(image))
            print('Length of sorted test image labels: ', len(labels))

        ### Added to gather data for clinic model training and validation###

        if self.mode == 'train':
            print('\tTraining data size: ', len(image))
            print('\tTraining label size: ', len(labels))
            self.list = []
            for i in range(len(image)):
                sub_list = []
                sub_list.append(image[i])
                sub_list.append(labels[i])
                self.list.append(tuple(sub_list))

        elif self.mode == 'val':
            print('\tValidation data size: ', len(image))
            print('\tValidation label size: ', len(labels))
            self.list = []
            for i in range(len(image)):
                sub_list = []
                sub_list.append(image[i])
                sub_list.append(labels[i])
                self.list.append(tuple(sub_list))

        elif self.mode == 'test':
            print('\tTest data size: ', len(image))
            print('\tTest label size: ', len(labels))
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

        # note to Mike: will probably have to add in self.augmentation to code because I dont see it implemented

        f_img, f_label = self.list[index]
        assert f_img.split('/')[-1].split('.')[0].split('_')[0] == f_label.split('/')[-1].split('.')[0].split('_')[
            0], 'check that mrns are the same'

        img, lab = np.load(f_img), np.load(f_label)
        # print('image size ', np.shape(img))

        # clinic_image_eval is for combined image and clinic model
        if not self.clinic_image_eval:
            if self.mode == 'train' and self.augmentation.lower() == 'true':
                img, lab = self.transform(img, lab)

            else:
                pass

            ###FILTER##
            if self.args.filter_input.lower() == 'true':
                pet = np.load(os.path.join(self.args.main_rd_data_path,
                                           f_img.split('/')[-1].split('.')[0].split('_')[
                                               0] + '_(130, 130, 92, 2).npy'))[..., 1]

                for i in range(self.args.in_modality):
                    img[..., i] = np.multiply(dilate_mask(pet=pet,
                                                          pet_cutoff=self.args.filter_pet_cutoff,
                                                          dilate_mm=self.args.filter_dilate_mm,
                                                          brain_cutoff=85, assert_z_length=92), img[..., i])
            else:
                pass
            ###FILTER##

            # cbct_ctsim_dose_image_review(img[..., 0], img[..., 1], img[..., 1],
            #                              save_fig_name=self.mode, view=False,
            #                              fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/')

            img = torch.FloatTensor(img)
            # Need to confirm w/ Kai but I believe this adjust the x,y,z, channel axis
            img = img.permute(3, 2, 0, 1)

            return img, lab

        else:
            if self.mode == 'train' and self.augmentation.lower() == 'true':
                img, lab = self.transform(img, lab)
            else:
                pass

            ###FILTER##
            if self.args.filter_input.lower() == 'true':
                pet = np.load(os.path.join(self.args.main_rd_data_path,
                                           f_img.split('/')[-1].split('.')[0].split('_')[
                                               0] + '_(130, 130, 92, 2).npy'))[..., 1]

                # old_pet

                for i in range(self.args.in_modality):
                    img[..., i] = np.multiply(dilate_mask(pet=pet,
                                                          pet_cutoff=self.args.filter_pet_cutoff,
                                                          dilate_mm=self.args.filter_dilate_mm,
                                                          brain_cutoff=85, assert_z_length=92), img[..., i])
            else:
                pass
            ###FILTER##

            # cbct_ctsim_dose_image_review(img[..., 0], img[..., 1], img[..., 1],
            #                              save_fig_name=self.mode, view=False,
            #                              fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/')

            img = torch.FloatTensor(img)
            # Need to confirm w/ Kai but I believe this adjust the x,y,z, channel axis
            img = img.permute(3, 2, 0, 1)

            return img, lab, f_img.split('/')[-1].split('_')[0]
