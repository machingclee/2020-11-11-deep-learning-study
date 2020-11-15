from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
                help="path to weights directory")
args = vars(ap.parse_args())

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, numOfClasses=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# the weight file will simply be replaced by one with smaller val_loss if no template string exists.
fname = os.path.sep.join([
    args["weights"],
    "weights-{epoch:03d}-{val_loss:.4f}.hdf5"
])

# Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
checkpointCallback = ModelCheckpoint(fname, monitor="val_loss", mode="min",
                                     save_best_only=True, verbose=2
                                     )
callbacks = [checkpointCallback]

model.fit(trainX, trainY, validation_data=(testX, testY),
          batch_size=64, epochs=40, callbacks=callbacks, verbose=1)
