from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras import callbacks
from pyimagesearch.nn.conv import MiniVGGNet
from pyimagesearch.callbacks import TrainingMonitorCallback
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output directory")
args = vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid))


((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, numOfClasses=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitorCallback(figPath=figPath, jsonPath=jsonPath)]

print("[INFO] training network ...")

model.fit(trainX, trainY, validation_data=(testX, testY),
          batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
