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
from tensorflow.keras.models import load_model

import tensorflow.keras.backend as K
import argparse
import json
import matplotlib
matplotlib.use("Agg")

CHECKPOINT_DIR = os.path.sep.join(["output", "checkpoints"])
OLD_MODEL_PATH = None
EPOCH_TO_START = 0
LEARNING_RATE = 1e-3

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


if OLD_MODEL_PATH is None:
    print("[INFO] compiling model...")
    model = DeeperGoogLeNet.build(width=64, height=64, depth=3, n_classes=config.NUM_CLASSES, reg=0.0002)
    opt = Adam(LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    print("[INFO] loading {}...".format(OLD_MODEL_PATH))
    model = load_model(OLD_MODEL_PATH)
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

callbacks = [EpochCheckpoint(config.CHECKPOINT_DIR),
             TrainingMonitorCallback(config.FIG_PATH, jsonPath=config.JSON_PATH)]

model.fit(train_generator.generator(),
          steps_per_epoch=train_generator.numOfImages//64,
          validation_data=val_generator.generator(),
          validation_steps=val_generator.numOfImages//64,
          epochs=10,
          max_queue_size=10,
          callbacks=callbacks,
          verbose=1)

train_generator.close()
val_generator.close()
