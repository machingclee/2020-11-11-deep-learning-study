import os

homeDirPath = os.path.dirname(__file__).replace("/config", "")

IMAGES_PATH = homeDirPath + "/dataset/kaggle_dogs_vs_cats/train"

NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

TRAIN_HDF5 = homeDirPath + "/dataset/kaggle_dogs_vs_cats/hdf5/train.hdf5"

VAL_HDF5 = homeDirPath + "/dataset/kaggle_dogs_vs_cats/hdf5/val.hdf5"
TEST_HDF5 = homeDirPath + "/dataset/kaggle_dogs_vs_cats/hdf5/test.hdf5"

MODEL_PATH = homeDirPath + "/models/alexnet_dogs_vs_cats.model"

DATASET_MEAN_JSON = homeDirPath + "/outputs/dogs_v_cats_mean.json"

OUTPUT_PATH = homeDirPath + "/outputs"
