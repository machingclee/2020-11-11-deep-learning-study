# usage: python test_ensemble.py --models models/lr_0.01
from pyimagesearch.utils.generate_classification_report_from_predictions import generate_classification_report_from_predictions
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import numpy as np

import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
                help="path to models directory")
args = vars(ap.parse_args())

(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0

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

labelBinarizer = LabelBinarizer()
testY = labelBinarizer.fit_transform(testY)

modelPaths = os.path.sep.join([args["models"], "*.model"])
modelPaths = glob.glob(modelPaths)
models = []

for (i, modelPath) in enumerate(modelPaths):
    print("[INFO] loading model{}/{}".format(i+1, len(modelPaths)))
    # modelPath is not necessary an hdf5 file
    models.append(load_model(modelPath))

predictions = []

for model in models:
    _prediction = model.predict(testX, batch_size=64)
    predictions.append(_prediction)

predictions = np.average(predictions, axis=0)
report = generate_classification_report_from_predictions(predictions,
                                                         testY,
                                                         labelNames=labelNames)
print(report)
