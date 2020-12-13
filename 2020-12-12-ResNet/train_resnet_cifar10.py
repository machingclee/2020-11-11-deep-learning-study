from numpy.core.defchararray import title
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitorCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
import os
import math


# connecting (0, 1e-1), (70, 1e-2), (100,0)
def polygonal_decay(x):
    if 0 <= x and x <= 70:
        return 1/10. - 9 * x/7000.
    else:
        return (100.-x)/3000.


lr_decay_callback = LearningRateScheduler(polygonal_decay)


checkpoint_dir = os.path.sep.join(["output", "checkpoints"])
work_title = "ResNet-Cifar-10"
max_epoch = 100

version = 4
start_at_epoch = 0
lr = 1e-1

prev_model_path = None
# prev_model_path = os.path.sep.join(["output", "checkpoints", "ResNet-Cifar-10-3-epoch-90.hdf5"])

new_version = 3
new_start_at_epoch = 90
new_lr = 1e-3

if prev_model_path is not None:
    version = new_version
    start_at_epoch = new_start_at_epoch
    lr = new_lr

config_path_with_version = os.path.sep.join(["output", work_title + "-"+str(version) + "-" + "training.png"])
json_path_with_version = os.path.sep.join(["output", work_title + "-"+str(version) + "-" + "training.json"])


((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis=0)
trainX = trainX - mean
testX = testX - mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

model = None

if prev_model_path is None:
    opt = SGD(lr=lr, momentum=0.9)
    model = ResNet.build(32, 32, 3, 10, (None, 9, 9, 9), (64, 64, 128, 256), reg=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    model = load_model(prev_model_path)
    print("[INFO] version: {}, start at epoch: {}".format(version, start_at_epoch))
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, new_lr)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

callbacks = [EpochCheckpoint(checkpoint_dir, model_title=work_title + "-" + str(version) + "-", startAt=start_at_epoch),
             TrainingMonitorCallback(config_path_with_version, jsonPath=json_path_with_version, startAt=start_at_epoch)]

if lr_decay_callback is not None:
    callbacks.append(lr_decay_callback)

model.fit(aug.flow(trainX, trainY, batch_size=128),
          validation_data=(testX, testY),
          steps_per_epoch=len(trainX)//128,
          epochs=max_epoch,
          callbacks=callbacks,
          verbose=1)
