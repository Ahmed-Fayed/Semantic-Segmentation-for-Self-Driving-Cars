# import the necessary packages
import torch
import os
from pathlib import Path


model_name = "UNet.pt"
ROOT_DIR = "D:/Software/CV_Projects/City_Segmentation"  # os.path.dirname(os.path.abspath(__file__))
# base path of the dataset
# DATASET_PATH = "D:/Software/CV_Projects/City_Segmentation/archive_2"
DATASET_PATH = os.path.join(ROOT_DIR, 'archive_2')

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 13
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
learning_rate = 0.001
# Set the gamma parameter for the exponential decay
gamma = 0.1  # Adjust as needed
EPOCHS = 10
BATCH_SIZE = 8

# define the input image dimensions
image_size = (160, 240)
PATCH_SIZE = 224

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
# BASE_OUTPUT = os.path.join(ROOT_DIR, 'Utils/output')
BASE_OUTPUT = ROOT_DIR + '/Utils/output'
Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)

# define the path to the artifacts output directory
# ARTIFACTS_OUTPUT = os.path.join(BASE_OUTPUT, "artifacts")
ARTIFACTS_OUTPUT = BASE_OUTPUT + "/artifacts/"
Path(ARTIFACTS_OUTPUT).mkdir(parents=True, exist_ok=True)

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(ARTIFACTS_OUTPUT, "unet_tgs_salt.pth")
METRIC_PLOT_PATH = os.path.join(ARTIFACTS_OUTPUT, "plot.png")
TEST_PATHS = os.path.join(ARTIFACTS_OUTPUT, "test_paths.txt")

""" Dataset Configs"""
train_ratio = 0.8
SEED = 1234
valid_ratio = 0.15
TEST_SPLIT = 0.1


""" MLFlow commands"""
# mlflow server --backend-store-uri sqlite:///backend.db