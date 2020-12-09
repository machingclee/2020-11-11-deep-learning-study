import os


def path_join(*arr):
    return os.path.sep.join([*arr])


def cd(path):
    return os.system("cd {}".format(path))


def open(path):
    os.system("start {}".format(path))


PROJECT_DIR = os.path.sep.join(os.path.dirname(__file__).split(os.path.sep)[:-1])
DATASET_DIR = path_join("d:", "datasets", "tiny-imagenet-200")

TRAIN_IMAGES_DIR = path_join(DATASET_DIR, "train")
VAL_IMAGES_DIR = path_join(DATASET_DIR, "val")
VAL_MAPPING_TXT = path_join(DATASET_DIR, "val", "val_annotations.txt")
WORDNET_IDS = path_join(DATASET_DIR, "wnids.txt")
WORD_LABELS = path_join(DATASET_DIR, "wnids.txt")


NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

TRAIN_HDF5_PATH = path_join(PROJECT_DIR, "hdf5", "train.hdf5")
VAL_HDF5_PATH = path_join(PROJECT_DIR, "hdf5", "val.hdf5")
TEST_HDF5_PATH = path_join(PROJECT_DIR, "hdf5", "test.hdf5")

DATASET_MEAN = path_join(PROJECT_DIR, "output", "tiny-image-net-200-mean.json")
MODEL_PATH = path_join(PROJECT_DIR, "output", "checkpoints", "model_epoch_70.hdf5")
FIG_PATH = path_join(PROJECT_DIR, "output", "deepergooglenet_tinyimagenet.png")
JSON_PATH = path_join(PROJECT_DIR, "output", "deepergooglenet_tinyimagenet.json")
