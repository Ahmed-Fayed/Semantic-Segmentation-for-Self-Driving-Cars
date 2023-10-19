import numpy as np
from config import *
import os
import matplotlib.pyplot as plt
from config import ARTIFACTS_OUTPUT
from PIL import Image


# helper function for image visualization
def display(img, label, img_title, label_title, save_name="vis", save=True):
    """
    Plot images in one row
    """
    plt.figure(figsize=(12, 6))
    plt.xticks([]);
    plt.yticks([])

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(img_title)

    plt.subplot(1, 2, 2)
    plt.imshow(label)
    plt.title(label_title)

    if not os.path.exists(ARTIFACTS_OUTPUT):
        os.mkdir(ARTIFACTS_OUTPUT)

    if save:
        fig_path = os.path.join(ARTIFACTS_OUTPUT, save_name + ".jpg")
        plt.savefig(fig_path, bbox_inches='tight')

    plt.show()


def visualize(image, mask, original_image=None, original_mask=None, fig_name="", save=True):
    fontsize = 16

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(10, 10), squeeze=True)
        f.set_tight_layout(h_pad=5, w_pad=5)

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 12), squeeze=True)
        plt.tight_layout(pad=0.2, w_pad=1.0, h_pad=0.01)

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original Image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original Mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed Image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed Mask', fontsize=fontsize)
        plt.show()

    if save:
        fig_path = os.path.join(ARTIFACTS_OUTPUT, fig_name + ".jpg")
        plt.savefig(fig_path, facecolor='w', transparent=False, bbox_inches='tight', dpi=100)



def display_images(data, no_img_mask):
    images, masks = data
    read_imgs = [None] * no_img_mask
    read_masks = [None] * no_img_mask
    for i in range(no_img_mask):
        read_imgs[i] = Image.open(images[i])
        read_masks[i] = Image.open(masks[i]).convert("L")

    fig, axes = plt.subplots(no_img_mask, 2)
    for i in range(no_img_mask):
        axes[i][0].imshow(read_imgs[i])
        axes[i][1].imshow(read_masks[i])

    # Show the plot
    plt.show()


def plot_results(train_res, val_res, ylabel="Loss", save=False, fig_name="loss"):
    plt.figure(figsize=(8, 6))
    plt.plot(train_res)
    plt.plot(val_res)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend(['train_'+ylabel, 'val_'+ylabel])
    plt.title("Semantic_Segmentation "+ylabel)

    if save:
        fig_path = os.path.join(ARTIFACTS_OUTPUT, fig_name + ".jpg")
        plt.savefig(fig_path, facecolor='w', transparent=False, bbox_inches='tight', dpi=100)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



