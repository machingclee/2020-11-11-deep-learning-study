# usage: python finetune_flower17.py --dataset dataset/flowers17 --model flowerws17.hdf5

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.loader import DatasetLoader
from pyimagesearch.nn.conv import FCHeadNet_FromFlatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from imutils import paths

import numpy as np
import argparse
import os

########################################
### parse argument from command line ###
########################################

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

############################
### load data and labels ###
############################

imagePaths = list(paths.list_images(args["dataset"]))
classNames = [path.split(os.path.sep)[-2] for path in imagePaths]
classNames = [str(className) for className in np.unique(classNames)]

# VGG network require images of shape (224, 224)
aspectAwarePreprocessor = AspectAwarePreprocessor(224, 224)
imageToArrayPreprocessor = ImageToArrayPreprocessor()

datasetLoader = DatasetLoader([aspectAwarePreprocessor,
                               imageToArrayPreprocessor])

(data, labels) = datasetLoader.load(imagePaths, verbose=500)

data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


##############################################
### define VGG base and concatenate models ###
##############################################

baseModel = VGG16(weights="imagenet",
                  include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

headModel = FCHeadNet_FromFlatten.build(baseModel, 256, len(classNames))

model = Model(inputs=baseModel.input, outputs=headModel)

################################
### start of warming up head ###
################################

# freeze model:
for layer in baseModel.layers:
    layer.trainable = False

opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# each step consumes batch_size number of training data.
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY),
                    epochs=25,
                    steps_per_epoch=len(trainX) // 32,
                    verbose=1)

predictions = model.predict(testX, batch_size=32)

print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames
                            ))

#################################
### start of finetuning model ###
#################################

# unfrezeze the layers and retrain the model:
for layer in baseModel.layers[15:]:
    layer.trainabla = True

print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=["accuracy"])

print("[INFO] re-training model...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY),
                    epochs=100,
                    steps_per_epoch=len(trainX) // 32,
                    verbose=1)

print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=32)

print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames
                            ))

model.save(args["model"])
