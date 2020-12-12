import os
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
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler

import tensorflow.keras.backend as K
import argparse
import json
import matplotlib
matplotlib.use("Agg")

# <========== training config settings ==========

# Everything (json, png, hdf5) in output folder is versioned by this value
VERSION = 6.1
MAX_NUM_EPOCHS = 40

# folders to save checkpoint directory
# callbacks objects do not accept absolute path:
CHECKPOINT_DIR = os.path.sep.join(["output", "checkpoints"])

# setting to train current model
INIT_LR = 1e-3
# adam | sgd
OPT_OPTION = "adam"
START_AT_EPOCH = 0

"""
Settings to retrieve previously trained model, copy and rename hdf5 + json to match version number 
if we want another new branch with adjusted hyper parameter.

Delete unwanted hdf5 file if we want to restart again at the head of the same branch.
"""
# PREV_MODEL_PATH = None
PREV_MODEL_PATH = os.path.sep.join([CHECKPOINT_DIR, "deeperGoogLenet-6.3-epoch-60.hdf5"])
# adam | sgd
NEW_VERSION = 6.3
NEW_INIT_LR = 1e-5
NEW_OPT_OPTION = "adam"
NEW_START_AT_EPOCH = 60

if PREV_MODEL_PATH is not None:
    VERSION = NEW_VERSION
    INIT_LR = NEW_INIT_LR
    OPT_OPTION = NEW_OPT_OPTION
    START_AT_EPOCH = NEW_START_AT_EPOCH


CONFIG_PATH_WITH_VERSION = os.path.sep.join([config.PROJECT_DIR, "output", "exp-"+str(VERSION)+"-"+"training.png"])
JSON_PATH_WITH_VERSION = os.path.sep.join([config.PROJECT_DIR, "output", "exp-"+str(VERSION)+"-"+"training.json"])


def poly_decay(epoch):
    return INIT_LR * ((1 - epoch/MAX_NUM_EPOCHS)**1)


opts = {
    "adam": Adam(INIT_LR),
    "sgd": SGD(learning_rate=INIT_LR, momentum=0.9)
}

opt = opts[OPT_OPTION]
lr_scheduler = LearningRateScheduler(poly_decay)

# ========== training config settings ==========>

aug = ImageDataGenerator(rotation_range=18,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

resize_pp = ResizePreprocessor(64, 64)
mean_subtraction_pp = MeanSubtractionPreprocessor(means["R"], means["G"], means["B"])
img_to_arr_pp = ImageToArrayPreprocessor()

train_generator = HDF5DatasetGenerator(config.TRAIN_HDF5_PATH, 64,
                                       aug=aug,
                                       preprocessors=[resize_pp, mean_subtraction_pp, img_to_arr_pp],
                                       n_classes=config.NUM_CLASSES)
val_generator = HDF5DatasetGenerator(config.VAL_HDF5_PATH, 64,
                                     aug=aug,
                                     preprocessors=[resize_pp, mean_subtraction_pp, img_to_arr_pp],
                                     n_classes=config.NUM_CLASSES)


if PREV_MODEL_PATH is None:
    print("[INFO] compiling model...")
    model = DeeperGoogLeNet.build(width=64, height=64, depth=3, n_classes=config.NUM_CLASSES, reg=0.0002)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    print("[INFO] loading {}...".format(PREV_MODEL_PATH))
    model = load_model(PREV_MODEL_PATH)
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, INIT_LR)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))


callbacks = [EpochCheckpoint(config.CHECKPOINT_DIR, model_title="deeperGoogLenet-"+str(VERSION)+"-", startAt=START_AT_EPOCH),
             TrainingMonitorCallback(CONFIG_PATH_WITH_VERSION, jsonPath=JSON_PATH_WITH_VERSION, startAt=START_AT_EPOCH)]

if lr_scheduler is not None:
    callbacks.append(lr_scheduler)

model.fit(train_generator.generator(),
          steps_per_epoch=train_generator.numOfImages//64,
          validation_data=val_generator.generator(),
          validation_steps=val_generator.numOfImages//64,
          epochs=MAX_NUM_EPOCHS,
          max_queue_size=10,
          callbacks=callbacks,
          verbose=1)

train_generator.close()
val_generator.close()
