from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import ResizePreprocessor
from pyimagesearch.datasets import DatasetLoader
from pyimagesearch.nn.conv import ShallowNet

from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to ouput model")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))


inputImageWidth = 32
inputImageHeight = 32


resizePreprocessor = ResizePreprocessor(
    width=inputImageWidth,    height=inputImageHeight
)
imageToArrayPreprocessor = ImageToArrayPreprocessor()

preprocessors = [
    resizePreprocessor,
    imageToArrayPreprocessor
]

datasetLoader = DatasetLoader(preprocessors=preprocessors)

# labels return ["dog", "cat", "dog", ...]
(data, labels) = datasetLoader.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42
)

# turn our list of strings into list of cannonical basis in R^n
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

optimizer = SGD(lr=0.005)

model = ShallowNet.build(width=inputImageWidth,
                         height=inputImageHeight,
                         depth=3,
                         numOfClasses=3)

model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])


H = model.fit(trainX, trainY,
              validation_data=(testX, testY),
              batch_size=32, epochs=100, verbose=1
              )
print("[INFO] saving networking ...")
model.save(args["model"])


print("[INFO] evaluating network ...")

# model.predict returns array of scores in the form:
# [[0.30036774 0.38575557 0.31387675], ...]

# input of predict expects array of np images, cannot be simply the np image, for testing, we need at least testX[0:1] to "keep dimension".
predictions = model.predict(testX)
print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=["cat", "dog", "panda"]
))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
