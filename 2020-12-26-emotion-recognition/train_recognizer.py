# usage: python train_recognizer.py -c checkpoints
# usage: python train_recognizer.py -c checkpoints -m checkpoints/epoch-15.hdf5 -s 15 -l 1e-3

from config import emotion_config as config
from deeptools import callbacks
from deeptools.preprocessing import ImageToArrayPreprocessor
from deeptools.callbacks import EpochCheckpoint
from deeptools.callbacks import TrainingMonitorCallback
from deeptools.io import HDF5DatasetGenerator
from deeptools.nn.conv import EmotionVGGNet
from deeptools.utils import get_digit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from imutils import paths
import re
import tensorflow.keras.backend as K
import argparse
import os
import shutil


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to specific model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
ap.add_argument("-l", "--learning-rate", type=float, default=1e-2, help="learning rate to restart")

args = vars(ap.parse_args())


train_aug = ImageDataGenerator(rotation_range=10,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               rescale=1/255.0,
                               fill_mode="nearest")

val_aug = ImageDataGenerator(rescale=1/255.0)
img_to_array_pp = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5,
                                config.BATCH_SIZE,
                                aug=train_aug,
                                preprocessors=[img_to_array_pp],
                                n_classes=config.NUM_CLASSES)

valGen = HDF5DatasetGenerator(config.VAL_HDF5,
                              config.BATCH_SIZE,
                              aug=val_aug,
                              preprocessors=[img_to_array_pp],
                              n_classes=config.NUM_CLASSES)

model = None
opt = None

figPath = os.path.sep.join([config.OUTPUT_DIR, "vggnet_emotion-0.png"])
jsonPath = os.path.sep.join([config.OUTPUT_DIR, "vggnet_emotion-0.json"])

if args["model"] is None:
    model = EmotionVGGNet.build(width=48, height=48, depth=1, n_classes=config.NUM_CLASSES)
    opt = Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))

    K.set_value(model.optimizer.lr, args["learning_rate"])
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

    startEpoch = str(args["start_epoch"])
    new_figPath = os.path.sep.join([config.OUTPUT_DIR, "vggnet_emotion-{}.png".format(startEpoch)])
    new_jsonPath = os.path.sep.join([config.OUTPUT_DIR, "vggnet_emotion-{}.json".format(startEpoch)])

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
    LearningRateScheduler(lambda epoch: 1e-3*(1-epoch/100.0))
]

model.fit(
    trainGen.generator(),
    steps_per_epoch=trainGen.numOfImages // config.BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numOfImages//config.BATCH_SIZE,
    epochs=100,
    max_queue_size=config.BATCH_SIZE*2,
    callbacks=callbacks,
    verbose=1
)

trainGen.close()
valGen.close()
