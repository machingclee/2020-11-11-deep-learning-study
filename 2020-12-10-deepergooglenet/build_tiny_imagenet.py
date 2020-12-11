from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.io import HDF5DatasetWriter
from pyimagesearch.utils import generate_progressbar
from pyimagesearch.utils import dataset_to_hdf5
from imutils import paths
import numpy as np
import os


trainPaths = list(paths.list_images(config.TRAIN_IMAGES_DIR))
trainLabels = [path.split(os.path.sep)[-3] for path in trainPaths]

# LabelEncoder turn our arr of classes into arr of integers, i.e., arr of binarized labels.
# LabelBiniarizer turn our arr of classes into arr of probability vectors, each has only 1 nonzero entry.

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES,
                         stratify=trainLabels, random_state=42)

(trainPaths, testPaths, trainLabels, testLabels) = split

mapping = open(config.VAL_MAPPING_TXT).read().strip().split("\n")
mapping = [row.split("\t")[:2] for row in mapping]
valPaths = [os.path.sep.join([config.DATASET_DIR, "val", "images", imgFile]) for (imgFile, _) in mapping]
valLabels = le.transform([label for (_, label) in mapping])


dataset_to_hdf5("train", trainPaths, trainLabels, config.TRAIN_HDF5_PATH, config.DATASET_MEAN)
dataset_to_hdf5("", valPaths, valLabels, config.VAL_HDF5_PATH)
dataset_to_hdf5("", testPaths, testLabels, config.TEST_HDF5_PATH)
