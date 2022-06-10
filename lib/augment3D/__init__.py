import random

import numpy as np
import torch
import random

from .elastic_deform import ElasticTransform
from .random_crop import RandomCropToLabels
from .random_flip import RandomFlip
from .random_rescale import RandomZoom
from .random_rotate import RandomRotation
from .random_shift import RandomShift, RandomShift_tta
from .gaussian_noise import GaussianNoise

functions = ['elastic_deform', 'random_crop', 'random_flip', 'random_rescale', 'random_rotate', 'random_shift']


class RandomChoice(object):
    """
    choose a random transform from list an apply
    transforms: transform to apply
    p: probability
    """

    def __init__(self, transforms=[],
                 p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors, label):
        augment = np.random.random(1) < self.p
        # print("augment is {}".format(augment))
        if not augment:
            return img_tensors, label
            
        t = random.choice(self.transforms)
        # print(img_tensors.ndim)
        if img_tensors.ndim==3:
            # print('it got here')
            img_tensors, label = t(img_tensors, label)
        else:
            for i in range(np.shape(img_tensors)[-1]):
                # print('length of img tensor: ', len(img_tensors))
                # print('shape of tensor: ', np.shape(img_tensors[...]))
                # print('shape of tensor: ', np.shape(img_tensors[..., i]))
                if i == 0:
                    ### do only once the augmentation to the label
                    img_tensors[..., i], label = t(img_tensors[..., i], label)
                else:
                    img_tensors[..., i], _ = t(img_tensors[..., i], label)
        return img_tensors, label


class ComposeTransforms(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms=[],
                 p=0.9):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors, label):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors, label

        for i in range(len(img_tensors)):

            for t in self.transforms:
                if i == (len(img_tensors) - 1):
                    ### do only once augmentation to the label
                    img_tensors[i], label = t(img_tensors[i], label)
                else:
                    img_tensors[i], _ = t(img_tensors[i], label)
        return img_tensors, label

class ComposeTransforms_tta(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms=[],
                 p=0.9):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors, label):
        # augment = np.random.random(1) < self.p
        # if not augment:
        #     return img_tensors, label
        # print('length of image tensor: ', np.shape(img_tensors)[-1])
        selected_transforms = random.sample(self.transforms, random.randint(1, 3))
        # print('selected transforms: ', selected_transforms)

        for i in range(np.shape(img_tensors)[-1]):

            # for t in self.transforms:
            for t in selected_transforms:
                if i == (len(img_tensors) - 1):
                    ### do only once augmentation to the label
                    img_tensors[..., i], label = t(img_tensors[..., i], label)
                else:
                    img_tensors[..., i], _ = t(img_tensors[..., i], label)
        return img_tensors, label