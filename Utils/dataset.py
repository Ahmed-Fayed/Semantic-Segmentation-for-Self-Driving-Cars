import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import cv2
from PIL import Image
from patchify import patchify
from glob import glob
import shutil
import copy
import random

from config import *
from Utils.utils import display, visualize, display_images

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as tf
import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy.ndimage as ndimage


""" Create train and test dataset"""
images = sorted(glob(DATASET_PATH + '/data*/data*/CameraRGB/*.png'))
masks = sorted(glob(DATASET_PATH + '/data*/data*/CameraSeg/*.png'))

print(DATASET_PATH)
print(images[0])
print(masks[0])
print(f"images size: {len(images)}")
print(f"masks size: {len(masks)}")

# display_images((images, masks), 3)


class lyft_dataset(Dataset):
    def __init__(self, images, masks, is_transform=True, augment=False):
        self.images = images
        self.masks = masks
        self.is_transform = is_transform
        self.augment = augment

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        if self.is_transform:
            img, mask = self.transform(img, mask)
        return img, mask

    def transform(self, img, mask):
        # resize
        resize = transforms.Resize(size=image_size)
        img = resize(img)
        mask = resize(mask)

        if self.augment:
            # Random horizontal flipping
            if random.random() > 0.5:
                img = tf.hflip(img)
                mask = tf.hflip(mask)
            # Random vertical Flipping
            if random.random() > 0.5:
                img = tf.vflip(img)
                mask = tf.vflip(mask)

        # Transform to tensor
        img = np.array(img)
        mask = np.array(mask)
        img = torch.from_numpy(img).float()
        img /= 255
        img = img.movedim(2, 0)
        mask = torch.from_numpy(mask).long()
        return img, mask

    def __len__(self):
        return len(self.images)


train_data, test_data, train_labels, test_labels = train_test_split(images, masks, test_size=0.2,
                                                                    shuffle=True, random_state=SEED)

train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                      shuffle=True, random_state=SEED)

print(f"train_data: {train_data[5]}")
print(f"train_labels: {train_labels[5]}")
print(f"valid_data: {valid_data[5]}")
print(f"valid_labels: {valid_labels[5]}")
print(f"test_data: {test_data[5]}")
print(f"test_labels: {test_labels[5]}")


t1 = A.Compose([
    A.Resize(160, 240),
    ToTensorV2()
])

train_dataset = lyft_dataset(train_data, train_labels, is_transform=True, augment=True)
valid_dataset = lyft_dataset(valid_data, valid_labels, is_transform=True, augment=False)
test_dataset = lyft_dataset(test_data, test_labels, is_transform=True, augment=False)

# Create a data loader
train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


# Iterate through the data loader
print(type(train_iterator))
print(train_iterator)
images, labels = next(iter(train_iterator))
print("Batch Size:", images.size(0))
print(f"image: {images[0]}, Image Shape: {images[0].size()}")
print(f"label: {labels[0]}, Label Shape: {labels[0].size()}")

print(f'num of training examples: {len(train_data)}')
print(f'num of validation examples: {len(valid_data)}')
print(f'num of testing examples: {len(test_data)}')
