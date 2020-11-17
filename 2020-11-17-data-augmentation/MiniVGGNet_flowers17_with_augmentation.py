from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))
classNames = [imPath.split(os.path.sep)[-2] for imPath in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aspectAwarePreprocess = AspectAwarePreprocessor(64, 64)

sd = SimpleDatasetLoader(preprocessors=[aspectAwarePreprocess])

(data, labels) = sd.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")


opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3,
                         numOfClasses=len(classNames))

model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                        validation_data=(testX, testY),
                        epochs=100, verbose=1)

predictions = model.predict(testX, batch_size=32)
print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=classNames
))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs Number")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./")
