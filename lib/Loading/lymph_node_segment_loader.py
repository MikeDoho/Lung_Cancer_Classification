# Python Modules
import os
import glob
import time
import numpy as np
import pandas as pd
import datetime

# Torch Modules
import torch
from torch.utils.data import Dataset
# Personal Modules
import lib.augment3D as augment3D
from lib.utils.logger import log
from lib.utils.evaluation_metrics import ct_mask_image_review
from lib.utils.general import pad_subtract_2d

# 2 D augmentation - not used yet
from torchvision.transforms import RandomRotation, GaussianBlur, RandomAffine


class LN_SEGMENT_LOAD(Dataset):
    """
    """

    def __init__(self, args, mode, dataset_path='./datasets_main', label_path='./datasets', classes=1,
                 exclude_mrns=[], train_path='./datasets_train', val_path='./datasets_val', test_path='./datasets_test',
                 exclude_blanks=0.3):
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
        self.exclude_blanks = self.args.exclude_blanks

        # self.aug_percent = aug_percent

        # self.train_mrn_list = train_mrn_list
        # self.val_mrn_list = val_mrn_list
        # self.test_mrn_list = test_mrn_list

        if self.augmentation.lower() == 'true' and self.mode == 'train':
            self.transform = augment3D.RandomChoice2d(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.05),
                            augment3D.RandomRotation2d_seg(),
                            augment3D.RandomShift2d_seg()],
                p=self.args.aug_percent)
            # self.transform = augment3D.RandomChoice(
            #     transforms=[augment3D.GaussianNoise(mean=0, std=0.1),
            #                 RandomAffine(degrees=0, translate=(.3, .6)),
            #                 RandomRotation(degrees=(45, 90))], p=self.args.aug_percent)

        # print(self.root)
        # image = glob.glob(os.path.join(self.root, '*.npy'))
        train_image = glob.glob(os.path.join(self.train_path, '*.npy'))
        # val_image = glob.glob(os.path.join(self.val_path, '*.npy'))
        # test_image = glob.glob(os.path.join(self.test_path, '*.npy'))
        # total_image = glob.glob(os.path.join(self.root, '*.npy'))

        # image = train_image + val_image + test_image
        image = train_image
        print('len of all image directories: ', len(image))
        image = [x for x in image if x.split('/')[-1].split('_')[0] not in self.exclude_mrns]

        print('len of all image directories after excluding mrns: ', len(image))
        labels = glob.glob(os.path.join(self.label_path, '*.npy'))
        labels = [x for x in labels if x.split('/')[-1].split('_')[0] not in self.exclude_mrns]

        if self.exclude_blanks > 0:
            image = sorted(image,
                           key=lambda x: (x.split('/')[-1].split('_')[0],
                                          x.split('/')[-1].split('_')[-2]))
            labels = sorted(labels,
                            key=lambda x: (x.split('/')[-1].split('_')[0],
                                           x.split('/')[-1].split('_')[-2]))

            zip_mrn_label = list(zip(image, labels))
            rand_value_gen = [np.random.randint(0, 100) for _ in range(len(image))]
            # print(rand_value_gen)
            print(f"length of random number generator: {len(rand_value_gen)}")

            # labels = [x for k, x in zip_mrn_label if np.max(np.load(x)) == 1 or
            #               (np.max(np.load(x)) == 1 and np.random.randint(0, 100) > 100*self.exclude_blanks)]
            # image = [k for k, x in zip_mrn_label if np.max(np.load(x)) == 1 or
            #               (np.max(np.load(x)) == 1 and np.random.randint(0, 100) > 100*self.exclude_blanks)]

            labels = [zip_list[1] for i, zip_list in enumerate(zip_mrn_label) if np.max(np.load(zip_list[1])) == 1 or
                      (np.max(np.load(zip_list[1])) == 0 and rand_value_gen[i] > 100 * self.exclude_blanks)]

            image = [zip_list[0] for i, zip_list in enumerate(zip_mrn_label) if np.max(np.load(zip_list[1])) == 1 or
                     (np.max(np.load(zip_list[1])) == 0 and rand_value_gen[i] > 100 * self.exclude_blanks)]

            # print(f"length of all image before sort {self.exclude_blanks}: {len(image)}")
            # print(f"length of all labels before sort {self.exclude_blanks}: {len(labels)}")

            # Resort files for safety
            image = sorted(image,
                           key=lambda x: (x.split('/')[-1].split('_')[0],
                                          x.split('/')[-1].split('_')[-2]))
            labels = sorted(labels,
                            key=lambda x: (x.split('/')[-1].split('_')[0],
                                           x.split('/')[-1].split('_')[-2]))

            print(f"length of all image after removing {self.exclude_blanks}: {len(image)}")
            print(f"length of all labels after removing {self.exclude_blanks}: {len(labels)}")
            print(image[0:20])
            print(labels[0:20])

        ### Added to gather data for clinic model training and validation###
        if self.mode == 'train':
            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            # image = [x for x in image if x.split('/')[-1].split('_')[0] in self.args.train_mrn_list]
            # labels = [x for x in labels if x.split('/')[-1].split('_')[0] in self.args.train_mrn_list]

            image = [x for x in image if tuple((x.split('/')[-1].split('_')[0], x.split('/')[-1].split('_')[-2])) in
                     self.args.train_zip_list]

            labels = [x for x in labels if tuple((x.split('/')[-1].split('_')[0], x.split('/')[-1].split('_')[-2])) in
                      self.args.train_zip_list]

            # image = sorted(list(set(image)),
            #                key=lambda x: (x.split('/')[-1].split('_')[0],
            #                               x.split('/')[-1].split('_')[-2]))
            #
            # labels = sorted(list(set(labels)),
            #                 key=lambda x: (x.split('/')[-1].split('_')[0],
            #                                x.split('/')[-1].split('_')[-2]))

            image = sorted(image,
                           key=lambda x: (x.split('/')[-1].split('_')[0],
                                          x.split('/')[-1].split('_')[-2]))

            labels = sorted(labels,
                            key=lambda x: (x.split('/')[-1].split('_')[0],
                                           x.split('/')[-1].split('_')[-2]))

            print('Length of training images: ', len(image))
            print('Length of training image labels: ', len(labels))
            print(image[0:20])
            print(labels[0:20])

        if self.mode == 'val':
            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            print('len val old: ', np.shape(image))
            # image = [x for x in image if x.split('/')[-1].split('_')[0] in self.args.val_mrn_list and \
            #          not x.split('/')[-1].split('.')[0].split('_')[-1].isnumeric()]
            # labels = [x for x in labels if x.split('/')[-1].split('_')[0] in self.args.val_mrn_list]

            image = [x for x in image if tuple((x.split('/')[-1].split('_')[0], x.split('/')[-1].split('_')[-2])) in
                     self.args.val_zip_list and not x.split('/')[-1].split('.')[0].split('_')[-1].isnumeric()]
            labels = [x for x in labels if tuple((x.split('/')[-1].split('_')[0], x.split('/')[-1].split('_')[-2])) in
                      self.args.val_zip_list]

            # image = sorted(list(set(image)), key=lambda x: x.split('/')[-1].split('_')[0])
            # image = sorted(list(set(image)),
            #                key=lambda x: (x.split('/')[-1].split('_')[0],
            #                               x.split('/')[-1].split('_')[-2]))
            #
            # labels = [x for x in labels if x.split('/')[-1].split('_')[0] in self.args.val_mrn_list]
            # # labels = sorted(list(set(labels)), key=lambda x: int(x.split('/')[-1].split('_')[0]))
            # labels = sorted(list(set(labels)),
            #                 key=lambda x: (x.split('/')[-1].split('_')[0],
            #                                x.split('/')[-1].split('_')[-2]))

            image = sorted(image,
                           key=lambda x: (x.split('/')[-1].split('_')[0],
                                          x.split('/')[-1].split('_')[-2]))

            # labels = sorted(list(set(labels)), key=lambda x: int(x.split('/')[-1].split('_')[0]))
            labels = sorted(labels,
                            key=lambda x: (x.split('/')[-1].split('_')[0],
                                           x.split('/')[-1].split('_')[-2]))

            print('Length of sorted validation images: ', len(image))
            print('Length of sorted validation image labels: ', len(labels))
            print(image[0:20])
            print(labels[0:20])

        if self.mode == 'test':
            # this will be needed for current code (need to take into account split in Train_Transfer_Clinic)
            print('Length of old test images: ', len(image))
            # image = [x for x in image if x.split('/')[-1].split('_')[0] in self.args.test_mrn_list and \
            #          not x.split('/')[-1].split('.')[0].split('_')[-1].isnumeric()]
            # labels = [x for x in labels if x.split('/')[-1].split('_')[0] in self.args.test_mrn_list]

            image = [x for x in image if tuple((x.split('/')[-1].split('_')[0], x.split('/')[-1].split('_')[-2])) in
                     self.args.test_zip_list and not x.split('/')[-1].split('.')[0].split('_')[-1].isnumeric()]
            labels = [x for x in labels if tuple((x.split('/')[-1].split('_')[0], x.split('/')[-1].split('_')[-2])) in
                      self.args.test_zip_list]

            # # image = sorted(list(set(image)), key=lambda x: x.split('/')[-1].split('_')[0])
            # image = sorted(list(set(image)),
            #                key=lambda x: (x.split('/')[-1].split('_')[0],
            #                               x.split('/')[-1].split('_')[-2]))
            #
            # labels = [x for x in labels if x.split('/')[-1].split('_')[0] in self.args.test_mrn_list]
            # # labels = sorted(list(set(labels)), key=lambda x: int(x.split('/')[-1].split('_')[0]))
            # labels = sorted(list(set(labels)),
            #                 key=lambda x: (x.split('/')[-1].split('_')[0],
            #                                x.split('/')[-1].split('_')[-2]))

            # image = sorted(list(set(image)), key=lambda x: x.split('/')[-1].split('_')[0])
            image = sorted(image,
                           key=lambda x: (x.split('/')[-1].split('_')[0],
                                          x.split('/')[-1].split('_')[-2]))

            # labels = sorted(list(set(labels)), key=lambda x: int(x.split('/')[-1].split('_')[0]))
            labels = sorted(labels,
                            key=lambda x: (x.split('/')[-1].split('_')[0],
                                           x.split('/')[-1].split('_')[-2]))

            print('Length of sorted test images: ', len(image))
            print('Length of sorted test image labels: ', len(labels))
            print(image[0:20])
            print(labels[0:20])

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

        # Loading image and label and confirming that input matches label match
        f_img, f_label = self.list[index]

        assert f_img.split('/')[-1].split('_')[0] == f_label.split('/')[-1].split('_')[0], \
            f"check that mrns are the same; input: {f_img.split('/')[-1].split('_')[0]}, " \
            f"label: {f_label.split('/')[-1].split('_')[0]}"

        assert f_img.split('/')[-1].split('_')[-2] == f_label.split('/')[-1].split('_')[-2], \
            f"check if slices are the same; input slice: {f_img.split('/')[-1].split('_')[-2]}, " \
            f"label slice: {f_label.split('/')[-1].split('_')[-2]}"

        # loading npy input and label
        img, lab = np.load(f_img), np.load(f_label)

        assert np.shape(img) == np.shape(lab), f"shapes are not matching; img: {np.shape(img)}, " \
                                               f"label: {np.shape(lab)}"

        if self.args.do_normalization.lower() == 'true':

            # Simple normalization
            img = np.clip(img, -1000, 2000)
            min_ = np.array(np.min(img), dtype=float)
            max_ = np.array(np.max(img), dtype=float)

            if not isinstance(min_, (int, float, np.int16, np.int32, np.ndarray)) or \
                    not isinstance(max_, (int, float, np.int16, np.int32, np.ndarray)):
                print(f_img)
                print(type(min_), type(max_))
                print('min: ', min_)
                print('max: ', max_)

            img = (img - min_) / (max_ - min_ + 1e-12)

            if lab.max() == 1 or lab.max() == 0:
                # pass
                if np.random.randint(0, 100) > 9990:
                    time_stamp = datetime.datetime.now()
                    ttime = time_stamp.strftime('%Y_%m_%d-%H_%M_%S')
                    ct_mask_image_review(img, lab,
                                         save_fig_name=f"norm_{self.args.short_note}_excludeblanks_{self.args.exclude_blanks}_"
                                                       f"augpercent_{self.args.aug_percent}_epoch_{self.args.n_epochs}_" + ttime + '.png',
                                         view=False,
                                         fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/')
            else:
                pass
        else:
            pass

        if self.mode == 'train' and self.augmentation.lower() == 'true':
            # print('')
            img_init = img.copy()
            lab_init = lab.copy()
            # print(np.shape(img), np.shape(img_init))
            img, lab = self.transform(img, lab)

            # to deal with the added dimensions w/ augmentation
            img = pad_subtract_2d(img, crop_shape=(self.args.input_H, self.args.input_W))
            lab = pad_subtract_2d(lab, crop_shape=(self.args.input_H, self.args.input_W))

            # if lab.max() == 1 or lab.max() == 0:
                # pass
            if np.random.randint(0, 10000) > 99995:
                # print('should see something')
                time_stamp = datetime.datetime.now()
                ttime = time_stamp.strftime('%Y_%m_%d-%H_%M_%S')
                ct_mask_image_review(img, img_init,
                                     save_fig_name=f"img_{self.mode}_{self.args.short_note}_excludeblanks_{self.args.exclude_blanks}_"
                                                   f"augpercent_{self.args.aug_percent}_epoch_{self.args.n_epochs}_" + ttime + '.png',
                                     view=False,
                                     fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/')
                ct_mask_image_review(lab, lab_init,
                                     save_fig_name=f"lab_{self.mode}_{self.args.short_note}_excludeblanks_{self.args.exclude_blanks}_"
                                                   f"augpercent_{self.args.aug_percent}_epoch_{self.args.n_epochs}_" + ttime + '.png',
                                     view=False,
                                     fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Residual_Disease/Pytorch/Data/saved_images/')

                # print('no error?')
                # else:
                #     pass

            else:
                pass

            # cbct_ctsim_dose_image_review(img[..., 0], img[..., 1], img[..., 2],
            #                              save_fig_name=self.mode+'_norm_aug', view=False,
            #                              fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/')

        else:
            pass

        img = np.expand_dims(img, axis=0)
        lab = np.expand_dims(lab, axis=0)

        img = torch.FloatTensor(img)
        lab = torch.FloatTensor(lab)

        return img, lab
