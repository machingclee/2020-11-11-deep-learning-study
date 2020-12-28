# usage: python train_model.py -c checkpoints
# usage: python train_model.py -c checkpoints -m checkpoints/epoch-15.hdf5 -s 15 -l 1e-4

from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dropout
from deeptools.utils import get_digit
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import ELU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from config import image_orientation_config as config
from deeptools.callbacks import TrainingMonitorCallback
from deeptools.callbacks import EpochCheckpoint
from tensorflow.keras import regularizers

import tensorflow.keras.backend as K
import argparse
import h5py
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to specific model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
ap.add_argument("-l", "--learning-rate", type=float, default=1e-2, help="learning rate to restart")

args = vars(ap.parse_args())


db = h5py.File(config.FEATURES_HDF5)
i = int(db["labels"].shape[0] * 0.75)
labels = to_categorical(db["labels"])
labels[1]


def build_logisticDenseModule():
    input = Input(shape=(512*7*7,))
    x = Dense(4096)(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)

    x = Dense(2048)(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.75)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.75)(x)

    x = Dense(4)(x)
    x = Activation("softmax")(x)

    return Model(input, x)


model = build_logisticDenseModule()

model = None
opt = None

figPath = os.path.sep.join([config.OUTPUT_DIR, "orientation-0.png"])
jsonPath = os.path.sep.join([config.OUTPUT_DIR, "orientation-0.json"])

if args["model"] is None:
    model = build_logisticDenseModule()
    opt = Adam(learning_rate=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))

    K.set_value(model.optimizer.lr, args["learning_rate"])
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

    # ===================================
    startEpoch = str(args["start_epoch"])
    new_figPath = os.path.sep.join([config.OUTPUT_DIR, "orientation-{}.png".format(startEpoch)])
    new_jsonPath = os.path.sep.join([config.OUTPUT_DIR, "orientation-{}.json".format(startEpoch)])

    prev_figPath = os.path.sep.join([
        config.OUTPUT_DIR,
        [file for file in os.listdir(config.OUTPUT_DIR) if ".png" in file
         and int(get_digit(file)) < int(startEpoch)][-1]
    ])

    prev_jsonPath = os.path.sep.join([
        config.OUTPUT_DIR,
        [file for file in os.listdir(config.OUTPUT_DIR) if ".json" in file
         and int(get_digit(file)) < int(startEpoch)][-1]
    ])

    shutil.copyfile(prev_figPath, new_figPath)
    shutil.copyfile(prev_jsonPath, new_jsonPath)

    figPath = new_figPath
    jsonPath = new_jsonPath


callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
    TrainingMonitorCallback(figPath, jsonPath=jsonPath, startAt=args["start_epoch"]),
    LearningRateScheduler(lambda epoch: 1e-3*(1-epoch/30.0))
]

model.fit(db["features"][:i],
          labels[:i],
          epochs=30,
          batch_size=32,
          validation_data=(db["features"][i:], labels[i:]),
          callbacks=callbacks,
          verbose=1)


# preds = model.predict(db["features"][i:])
# print(classification_report(labels.argmax(axis=1), preds.argmax(axis=1), target_names=db["label_names"]))

db.close()
