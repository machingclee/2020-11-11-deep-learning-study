from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import json
import cv2
import os


trainPaths = list(paths.list_images(config.TRAIN_IMAGES_DIR))
trainLabels = [path.split(os.path.sep)[-3] for path in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES,
                         stratify=trainLabels, random_state=42)

mapping = open(config.VAL_MAPPING_TXT).read().strip().split("\n")
mapping = [row.split("\t")[:2] for row in mapping]
valPath = [os.path.sep.join([config.DATASET_DIR, "val", "images", imgFile]) for (imgFile, _) in mapping]
valLabels = le.transform([label for (_, label) in mapping])


(R, G, B) = ([], [], [])

# to be continued ....
