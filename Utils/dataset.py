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


""" Spliting Dataset """


def split_dataset_atten(dataset_path, output_dir, verbose=True):
    dataset_images = sorted(glob(dataset_path + '/data*/data*/CameraRGB/*.png'))
    dataset_masks = sorted(glob(dataset_path + '/data*/data*/CameraSeg/*.png'))
    train_images, test_images, train_masks, test_masks = train_test_split(dataset_images, dataset_masks,
                                                                          test_size=TEST_SPLIT, shuffle=True, random_state=SEED)

    for train_img, train_mask in tqdm(zip(train_images, train_masks), total=len(train_images)):
        train_dir = os.path.join(output_dir, "train_dataset")
        train_imgs_dir = os.path.join(train_dir, "images")
        train_masks_dir = os.path.join(train_dir, "masks")

        Path(train_imgs_dir).mkdir(parents=True, exist_ok=True)
        Path(train_masks_dir).mkdir(parents=True, exist_ok=True)

        shutil.copyfile(train_img, os.path.join(train_imgs_dir, train_img.split("\\")[-1]))
        shutil.copyfile(train_mask, os.path.join(train_masks_dir, train_mask.split("\\")[-1]))

    for test_img, test_mask in tqdm(zip(test_images, test_masks), total=len(test_images)):
        test_dir = os.path.join(output_dir, "test_dataset")
        test_imgs_dir = os.path.join(test_dir, "images")
        test_masks_dir = os.path.join(test_dir, "masks")

        Path(test_imgs_dir).mkdir(parents=True, exist_ok=True)
        Path(test_masks_dir).mkdir(parents=True, exist_ok=True)

        shutil.copyfile(test_img, os.path.join(test_imgs_dir, test_img.split("\\")[-1]))
        shutil.copyfile(test_mask, os.path.join(test_masks_dir, test_mask.split("\\")[-1]))

    if verbose:
        print(f"dataset_images_attention size: {len(dataset_images)}")
        print(f"dataset_masks_attention size: {len(dataset_masks)}")

        print(f"train_images_atten size: {len(train_images)}")
        print(f"test_images_atten size: {len(test_images)}")
        print(f"train_masks_atten size: {len(train_masks)}")
        print(f"test_masks_atten size: {len(test_masks)}")
        print(train_images[0])

    return train_images, test_images, train_masks, test_masks


# split_dataset_atten(DATASET_PATH, BASE_OUTPUT, verbose=True)


""" Data Augmentation """
transform = A.Compose([
        # A.RandomCrop(width=width, height=height, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=[60, 300], p=1.0, interpolation=cv2.INTER_NEAREST),
        A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.3], contrast_limit=0.2, p=1.0),
        A.OneOf([
            A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5)], p=1.0),
    ], p=1.0)


def test_augmentation(img_path, mask_path, transform=None):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, 0)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    cv2.imwrite('./image.png', cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite('./mask.png', cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2RGB))

    visualize(transformed_image, transformed_mask, image, mask, "test_augmentation.jpg", save=True)


img_path = os.path.join(BASE_OUTPUT, "train_dataset", "images", "02_00_000.png")
mask_path = os.path.join(BASE_OUTPUT, "train_dataset", "masks", "02_00_000.png")
# test_augmentation(img_path, mask_path, transform)


def augment_dataset(imgs_dir, masks_dir, transform, count):
    '''Function for data augmentation
        Input:
            count - total no. of images after augmentation = initial no. of images * count
        Output:
            writes augmented images (input images & segmentation masks) to the working directory
    '''

    imgs_output_dir = os.path.join(BASE_OUTPUT, "train_dataset/aug_images")
    masks_output_dir = os.path.join(BASE_OUTPUT, "train_dataset/aug_masks")
    Path(imgs_output_dir).mkdir(parents=True, exist_ok=True)
    Path(masks_output_dir).mkdir(parents=True, exist_ok=True)

    i = 0
    for i in tqdm(range(count)):
        for img_name, mask_name in zip(sorted(os.listdir(imgs_dir)), sorted(os.listdir(masks_dir))):
            img_path = os.path.join(imgs_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask_path = os.path.join(masks_dir, mask_name)
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            transformed = transform(image=img, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            cv2.imwrite(imgs_output_dir + '/aug_{}_'.format(str(i + 1)) + img_name, transformed_image)
            cv2.imwrite(masks_output_dir + '/aug_{}_'.format(str(i + 1)) + img_name, transformed_mask)


imgs_dir = BASE_OUTPUT + "/train_dataset/images"
masks_dir = BASE_OUTPUT + "/train_dataset/masks"
# augment_dataset(imgs_dir, masks_dir, transform, count=2)


class LyftDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted(glob(images_dir + "/*.png"))
        self.masks = sorted(glob(masks_dir + "/*.png"))
        self.transform = transform

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        mask = np.array(Image.open(self.masks[idx]))
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
            mask = torch.max(mask, dim=2)[0]

        return image, mask

    def __len__(self):
        return len(self.images)


t1 = A.Compose([
    A.Resize(image_size[0], image_size[1]),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# images_dir = BASE_OUTPUT + "/train_dataset/aug_images"
# masks_dir = BASE_OUTPUT + "/train_dataset/aug_masks"
images_dir = BASE_OUTPUT + "/train_dataset/images"
masks_dir = BASE_OUTPUT + "/train_dataset/masks"
train_dataset = LyftDataset(images_dir, masks_dir, transform=t1)

print(f"train_dataset: {train_dataset.__len__()}")
val_num = int(valid_ratio * train_dataset.__len__())
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_dataset.__len__() - val_num, val_num],
                                                               generator=torch.Generator().manual_seed(SEED))

print(f"train_dataset: {train_dataset.__len__()}")
print(f"val num: {val_num}")
print(f"train_dataset: {train_dataset}")
# # Create a data loader
train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_iterator = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


# Iterate through the data loader
print(type(train_iterator))
print(train_iterator)
images, labels = next(iter(train_iterator))
print("Batch Size:", images.size(0))
print(f"image: {images[0]}, Image Shape: {images[0].size()}")
print(f"label: {labels[0]}, Label Shape: {labels[0].size()}")

print(f'num of training examples: {len(train_dataset)}')
print(f'num of validation examples: {len(val_dataset)}')

# for img,mask in train_iterator:
#     img1 = np.transpose(img[0,:,:,:],(1,2,0))
#     mask1 = np.array(mask[0,:,:])
#     img2 = np.transpose(img[1,:,:,:],(1,2,0))
#     mask2 = np.array(mask[1,:,:])
#     img3 = np.transpose(img[2,:,:,:],(1,2,0))
#     mask3 = np.array(mask[2,:,:])
#     fig , ax =  plt.subplots(3, 2, figsize=(18, 18))
#     ax[0][0].imshow(img1)
#     ax[0][1].imshow(mask1)
#     ax[1][0].imshow(img2)
#     ax[1][1].imshow(mask2)
#     ax[2][0].imshow(img3)
#     ax[2][1].imshow(mask3)
#     break
# plt.show()
