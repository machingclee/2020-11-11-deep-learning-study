# usage: python train_models.py --output output --models models
from pyimagesearch.utils.plot_training_graph import plot_training_graph
from pyimagesearch.utils.generate_classification_report import generate_classification_report
import matplotlib

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

######################
### parse argument ###
######################

ap = argparse.ArgumentParser()
ap.add_argument("-o",
                "--output",
                required=True,
                help="path to output directory")

ap.add_argument("-m",
                "--models",
                required=True,
                help="path to output models directory")

ap.add_argument("-n",
                "--num-models",
                type=int,
                default=5,
                help="numer of models to train")

args = vars(ap.parse_args())

####################
### load dataset ###
####################

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

labelBinarizer = LabelBinarizer()
trainY = labelBinarizer.fit_transform(trainY)
testY = labelBinarizer.transform(testY)

labelNames = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

##########################################
### instantiate augmentation generator ###
##########################################

aug = ImageDataGenerator(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest")

##############################################
### train each individual MiniVGGNet model ###
##############################################

for i in np.arange(0, args["num_models"]):
    print("[INFO] training model {}/{}".format(i+1, args["num_models"]))
    opt = SGD(lr=0.01,
              decay=0.01/40,
              momentum=0.9,
              nesterov=True)
    model = MiniVGGNet.build(width=32,
                             height=32,
                             depth=3,
                             numOfClasses=10)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    # H will be used later to plot loss-accuracy-against-epoch graph
    H = model.fit(aug.flow(trainX, trainY, batch_size=64),
                  validation_data=(testX, testY),
                  epochs=40,
                  steps_per_epoch=len(trainX) // 64,
                  verbose=1
                  )

    modelPath = [args["models"], "model_{}.model".format(i)]
    model.save(os.path.sep.join(modelPath))

#######################
### generate report ###
#######################

    report = generate_classification_report(
        model, testX, testY, batch_size=64, labelNames=labelNames)
    filePath = [args["output"], "model_{}.text".format(i+1)]

    fs = open(os.path.sep.join(filePath), "w")
    fs.write(report)
    fs.close()

    plot_training_graph(
        H,
        saveFilePath=os.path.sep.join(
            [args["output"], "model_{}.png".format(i+1)]),
        epochs=40,
        title="Training Loss and Accuracy for model{}".format(i+1),
    )
