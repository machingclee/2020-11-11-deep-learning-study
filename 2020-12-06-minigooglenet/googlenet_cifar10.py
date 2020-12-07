from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.engine.training import Model
from pyimagesearch.nn.conv import MiniGoogLeNet
from pyimagesearch.callbacks import TrainingMonitorCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10

import numpy as np
import argparse
import os


# CUDA_VISIBLE_DEVICES=-1
# CUDA_VISIBLE_DEVICES=0

NUM_EPOCHS = 70
INIT_LR = 5e-3

figPath = os.path.sep.join(["outputs", "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join(["outputs", "{}.json".format(os.getpid())])
modelPath = os.path.sep.join(["models", "minigooglenet_cifar10.hdf5"])


def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    x = epoch/float(maxEpochs)

    return baseLR * ((1-x) ** power)


print("[INFO] loading CIFAR-10 data")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# mean subtraction, preprocessors should apply equally to BOTH trainX and testX
mean = np.mean(trainX, axis=0)
trainX = trainX - mean
testX = testX - mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

callbacks = [TrainingMonitorCallback(figPath=figPath),
             LearningRateScheduler(poly_decay)]

opt = SGD(lr=INIT_LR, momentum=0.9)

model = MiniGoogLeNet.build(width=32, height=32, depth=3, numOfClasses=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(aug.flow(trainX, trainY, batch_size=64),
          validation_data=(testX, testY),
          steps_per_epoch=len(trainX) // 64,
          epochs=NUM_EPOCHS,
          callbacks=callbacks)

print("[INFO] serializing network ...")
model.save(modelPath)
