
from numpy.core.defchararray import title
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitorCallback
from pyimagesearch.preprocessing import ResizePreprocessor
from pyimagesearch.preprocessing import MeanSubtractionPreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from config import tinyimagenet_config as config
from astropy.utils.decorators import classproperty
import tensorflow.keras.backend as K
import numpy as np
import os
import json


class TrainingConfig:

    checkpoint_dir = os.path.sep.join(["output", "checkpoints"])
    work_title = "ResNet-TinyImagenet-200"
    max_epoch = 100

    _version = 1
    _start_at_epoch = 0
    _lr = 1e-1

    # prev_model_path = None
    prev_model_path = os.path.sep.join(["output", "checkpoints", "ResNet-TinyImagenet-200-1-epoch-25.hdf5"])
    _new_version = 1
    _new_start_at_epoch = 25
    _new_lr = 1e-1

    @classproperty
    def version(self):
        return self._version if self.prev_model_path is None else self._new_version

    @classproperty
    def start_epoc_at(self):
        return self._start_at_epoch if self.prev_model_path is None else self._new_start_at_epoch

    @classproperty
    def lr(self):
        return TrainingConfig._lr if TrainingConfig.prev_model_path is None else TrainingConfig._new_lr

    @classproperty
    def config_path_with_version(self):
        return os.path.sep.join(["output", self.work_title + "-"+str(self.version) + "-" + "training.png"])

    @classproperty
    def json_path_with_version(self):
        return os.path.sep.join(["output", self.work_title + "-"+str(self.version) + "-" + "training.json"])


max_epoch = TrainingConfig.max_epoch
prev_model_path = TrainingConfig.prev_model_path
checkpoint_dir = TrainingConfig.checkpoint_dir
work_title = TrainingConfig.work_title

config_path_with_version = TrainingConfig.config_path_with_version
json_path_with_version = TrainingConfig.json_path_with_version
version = TrainingConfig.version
start_at_epoch = TrainingConfig.start_epoc_at
lr = TrainingConfig.lr

max_epoch
prev_model_path
checkpoint_dir
work_title
config_path_with_version
json_path_with_version
version
start_at_epoch
lr


model = None

if prev_model_path is None:
    opt = SGD(lr=lr, momentum=0.9)
    model = ResNet.build(64, 64, 3, config.N_CLASSES, (None, 3, 4, 6), (64, 128, 256, 512), reg=0.0005, dataset="tiny_imagenet")
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    model = load_model(prev_model_path)
    print("[INFO] version: {}, start at epoch: {}".format(version, start_at_epoch))
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, lr)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

callbacks = [EpochCheckpoint(checkpoint_dir, model_title=work_title + "-" + str(version) + "-", startAt=start_at_epoch),
             TrainingMonitorCallback(config_path_with_version, jsonPath=json_path_with_version, startAt=start_at_epoch)]


aug = ImageDataGenerator(
    rotation_range=18,
    zoom_range=0.15,
    shear_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

means = json.loads(open(config.MEAN_JSON_PATH).read())
(mu_R, mu_G, mu_B) = (means["R"], means["G"], means["B"])

resize_pp = ResizePreprocessor(64, 64)
mean_sub_pp = MeanSubtractionPreprocessor(mu_R, mu_G, mu_B)
img_to_arr_pp = ImageToArrayPreprocessor()

pps = [resize_pp, mean_sub_pp, img_to_arr_pp]

train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5_PATH, preprocessors=pps, batchSize=64, aug=aug, n_classes=config.N_CLASSES)
val_gen = HDF5DatasetGenerator(config.VAL_HDF5_PATH, preprocessors=pps, batchSize=64, aug=aug, n_classes=config.N_CLASSES)

model.fit(train_gen.generator(),
          validation_data=val_gen.generator(),
          steps_per_epoch=train_gen.numOfImages//64,
          validation_steps=val_gen.numOfImages//64,
          epochs=max_epoch,
          callbacks=callbacks,
          max_queue_size=10,
          verbose=1)
