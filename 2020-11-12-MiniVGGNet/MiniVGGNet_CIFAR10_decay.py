from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.python.keras import callbacks
from pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

matplotlib.use("Agg")


def step_decay(initial_factor, decay_factor, decay_per_epoch):
    def decay_factor_at_epoach(epoch):
        alpha = initial_factor * \
            (decay_factor ** np.floor((1 + epoch)/decay_per_epoch))
        return alpha

    return decay_factor_at_epoach


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data ...")
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

callbacks = [LearningRateScheduler(
    step_decay(initial_factor=0.01, decay_factor=0.25, decay_per_epoch=5)
)]

opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, numOfClasses=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64, epochs=40, callbacks=callbacks, verbose=1)

print("[INFO] evaluating network ...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=labelNames
))

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
