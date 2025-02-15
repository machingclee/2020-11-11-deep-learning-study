from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

# stop the plotted image from popping-up:
matplotlib.use("Agg")

# argument parser routine:
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/acc plot")
args = vars(ap.parse_args())


def log(msg):
    print("[INFO] {}".format(msg))


# load and preprocess data:
log("loading CIFAR-10 data ...")
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

log("compiling model...")

# nesterov=True => we use Nestrov accerated gradient
# study the momentum term later
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, numOfClasses=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

log("training network ...")
H = model.fit(trainX, trainY,
              validation_data=(testX, testY),
              batch_size=64,
              epochs=40,
              verbose=1)

predictions = model.predict(testX, batch_size=64)
print(
    classification_report(testY.argmax(axis=1),
                          predictions.argmax(axis=1),
                          target_names=labelNames)
)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch Numbers")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
