# usage: python test_recognizer.py -m checkpoints/epoch-100.hdf5

from config import emotion_config as config
from deeptools.preprocessing import ImageToArrayPreprocessor
from deeptools.io import HDF5DatasetGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="path to model checkpoint to load")
args = vars(ap.parse_args())

test_aug = ImageDataGenerator(rescale=1/255.0)
img_to_arr_pp = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(
    config.TEST_HDF5,
    config.BATCH_SIZE,
    aug=test_aug,
    preprocessors=[img_to_arr_pp],
    n_classes=config.NUM_CLASSES
)

print("[INFO] loading {}".format(args["model"]))
model = load_model(args["model"])

(_, acc) = model.evaluate(
    testGen.generator(),
    steps=testGen.numOfImages // config.BATCH_SIZE,
    max_queue_size=config.BATCH_SIZE*2
)

print("[INFO] accuracy {:.2f}%".format(acc*100))

testGen.close()
