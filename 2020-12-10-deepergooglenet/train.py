from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import ResizePreprocessor
from pyimagesearch.preprocessing import MeanSubtractionPreprocessor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitorCallback
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import DeeperGoogLeNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import tensorflow.keras.backend as K
import argparse
import json
import matplotlib
matplotlib.use("Agg")

CHECKPOINT_DIR = ""
MODEL_OUTPUT_PATH = ""
EPOCH_TO_START = 0

aug = ImageDataGenerator(rotation=18,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

means = json.load(open(config.DATASET_MEAN).read())

resize_pp = ResizePreprocessor(64, 64)
mean_subtraction_pp = MeanSubtractionPreprocessor(means["R"], means["G"], means["B"])
img_to_arr_pp = ImageToArrayPreprocessor()

train_generator = HDF5DatasetGenerator(config.TRAIN_HDF5_PATH, 64,
                                       aug=aug,
                                       preprocessors=[resize_pp, mean_subtraction_pp, img_to_arr_pp])
