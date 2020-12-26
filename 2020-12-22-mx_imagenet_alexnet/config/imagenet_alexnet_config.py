import os

BASE_PATH = os.path.sep.join(["C:", "Users", "user", "Repos", "Python", "DeepLearning", "deep-learning-study", "datasets", "ILSVRC2012"])
IMAGES_DIR = os.path.sep.join([BASE_PATH, "Data", "CLS-LOC"])
IMAGES_SETS_DIR = os.path.sep.join([BASE_PATH, "ImageSets", "CLS-LOC"])
DEVKIT_DIR = os.path.sep.join([BASE_PATH, "devkit", "data"])

WORD_IDS = os.path.sep.join([DEVKIT_DIR, "map_clsloc.txt"])
TRAIN_LIST = os.path.sep.join([IMAGES_SETS_DIR, "train_cls.txt"])
VAL_LIST = os.path.sep.join([IMAGES_SETS_DIR, "val.txt"])
VAL_LABELS = os.path.sep.join([DEVKIT_DIR, "ILSVRC2015_clsloc_validation_ground_truth.txt"])

VAL_BLACKLIST = os.path.sep.join([DEVKIT_DIR, "ILSVRC2015_clsloc_validation_blacklist.txt"])

NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50 * NUM_CLASSES


TRAIN_MX_LIST = os.path.sep.join([BASE_PATH, "lists", "train.lst"])
VAL_MX_LIST = os.path.sep.join([BASE_PATH, "lists", "val.lst"])
TEST_MX_LIST = os.path.sep.join([BASE_PATH, "lists", "test.lst"])

TRAIN_MX_REC = os.path.sep.join([BASE_PATH, "rec", "train.rec"])
VAL_MX_REC = os.path.sep.join([BASE_PATH, "rec", "val.rec"])
TEST_MX_REC = os.path.sep.join([BASE_PATH, "rec", "test.rec"])

DATASET_MEAN = "output/imagenet_mean.json"

BATCH_SIZE = 128
NUM_DEVICES = 1
