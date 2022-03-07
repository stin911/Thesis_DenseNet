from __future__ import print_function, division
import pandas as pd
import warnings
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

warnings.filterwarnings("ignore")
import numpy as np
import SimpleITK as sitk
import torchio as tio
import visualize


def Jitter(mat):
    """

    :param mat: image numpy array
    :param factor:  factor brightness and contrast
    :return: image
    """
    mx = np.max(mat)
    factor = round(((mx * 0.1) / 100), 2)

    prob = random.uniform(0, 1)
    if prob > 0.50:
        mat = mat + factor

    """else:
        mat = mat * factor"""
    return mat


def rotate(image):
    """

    :param image: SimpleItk Image
    :return:
    """
    random_affine = tio.RandomAffine(center='origin', isotropic=False, scales=(1, 1), degrees=(0, 0, 0, 0, -10, 10),
                                     translation=(0, 0))
    image_affine = random_affine(image)
    return image_affine


def translate(image):
    """

    :param image:SimpleItk Image
    :return:
    """
    random_affine = tio.RandomAffine(center='origin', isotropic=False, scales=(1, 1), degrees=(0, 0, 0, 0, 0, 0),
                                     translation=(-10, 10, -10, 10, 0, 0))
    image_affine = random_affine(image)
    return image_affine


def clip_and_normalize(ct_slice, min_clip_value, max_clip_value):
    """
    This method clips ct slice between minimum and maximum values and normalize it between 0 and 1
    :param max_clip_value:
    :param min_clip_value:
    :param ct_slice: numpy img array
    :return: clipped and normalized img array
    """
    ct_slice_clip_norm = (ct_slice - min_clip_value) / (max_clip_value - min_clip_value)
    ct_slice_clip_norm[ct_slice_clip_norm < 0] = 0
    ct_slice_clip_norm[ct_slice_clip_norm > 1] = 1

    return ct_slice_clip_norm


class Bms(Dataset):
    """
    Class represent the dataset
    """

    def __init__(self, csv_file, root_dir, transform=None, aug=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dt = {'name': 'str', 'Pneumonitis': 'int'}
        self.csv = csv_file
        self.annotation = pd.read_csv(csv_file, sep=',', dtype=dt)
        self.root_dir = root_dir
        self.transform = transform
        self.aug = aug

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        """

        :param idx:
        :return: an element of dataset
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.root_dir + '_' + str(self.annotation.iloc[idx, 0]) + '_0.nrrd'

        image = sitk.ReadImage(path)
        # Apply data Augmentation
        prob = random.uniform(0, 1)
        if self.aug:

            if prob < 0.2:
                image = rotate(image)

            elif prob < 0.4:
                image = translate(image)

            elif prob < 0.6:
                target_shape = image.GetWidth() - int(image.GetWidth() * 5 / 100), image.GetHeight() - int(
                    image.GetHeight() * 5 / 100), image.GetDepth() - int(image.GetDepth() * 5 / 100)
                crop_pad = tio.CropOrPad(target_shape)
                image = crop_pad(image)

        # Crop the image or pad to the same dimension
        target_shape = 320, 320, 80
        crop_pad = tio.CropOrPad(target_shape)
        image = crop_pad(image)
        
        
        

        image = sitk.GetArrayFromImage(image)

        # ---------------------------------Contrast-Gamma-Augmentation
        if self.aug:

            # prob = random.uniform(0, 1)
            if 0.60 < prob < 0.80:
                image = Jitter(image)

        y = int(self.annotation.iloc[idx, 1])

        sample = {'image': image, 'label': y, 'name': self.annotation.iloc[idx, 0]}
        if self.transform:
            sample = self.transform(sample)

        return sample


def check_even(x):
    """
    :param x: int number
    :return:
    """
    if x % 2 == 1:
        x1 = int(((x - 1) / 2) + 1)
        x2 = int((x - 1) / 2)
        return x1, x2
    else:
        x1 = int(x / 2)
        x2 = int(x / 2)
        return x1, x2


class NormPad(object):

    def __call__(self, sample):
        """

        :param sample: dictionary with the image and the label
        :return: the dictionary with the image normalized and padded
        """

        arr = sample['image']
        # Normalize and clip
        arr = clip_and_normalize(arr, -1000, 0)

        return {'image': arr, 'label': sample['label'], 'name': sample['name']}


class ToTensor(object):
    def __call__(self, sample):
        """
        :param sample: dictionary image -label
        :return: dictionary with the image converted to tensor adn the label
        """
        arr = sample['image']

        image = torch.from_numpy(arr).float()

        image = image.unsqueeze(0)

        return {'image': image, 'label': sample['label'], 'name': sample['name']}



<<<<<<< Updated upstream
=======


>>>>>>> Stashed changes
