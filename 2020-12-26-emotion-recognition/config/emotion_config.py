import os

sep = os.path.sep

DATA_DIR = sep.join(["C:", "Users", "user", "Repos", "Python", "DeepLearning", "deep-learning-study", "datasets", "fer2013"])
PROJECT_DIR = sep.join(os.path.realpath(__file__).split(sep)[:-2])


INPUT_PATH = sep.join([DATA_DIR, "fer2013.csv"])

NUM_CLASSES = 6

TRAIN_HDF5 = sep.join([PROJECT_DIR, "hdf5", "train.hdf5"])
VAL_HDF5 = sep.join([PROJECT_DIR, "hdf5", "val.hdf5"])
TEST_HDF5 = sep.join([PROJECT_DIR, "hdf5", "test.hdf5"])

BATCH_SIZE = 128
OUTPUT_DIR = sep.join([PROJECT_DIR, "output"])
