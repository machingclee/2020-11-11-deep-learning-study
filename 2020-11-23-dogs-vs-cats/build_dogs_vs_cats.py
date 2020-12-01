# usage: python build_dogs_vs_cats.py
import enum
from logging import INFO
from config import dogs_vs_cats_config as config

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.io import HDF5DatasetWriter
from pyimagesearch.utils import generate_progressbar
from imutils import paths
import numpy as np

import json
import cv2
import os

###################################
### load image paths and labels ###
###################################

trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [path.split(os.path.sep)[-1].split(".")[0]
               for path in trainPaths]

labelEncoder = LabelEncoder()
# sorted and return integers
trainLabels = labelEncoder.fit_transform(trainLabels)

#######################################################
### split training set into validation and test set ###
#######################################################

# split a portion of training set into test set
# we use a 'proportion' as test_size in the past
"""
stratify means to retain the ratio of labels in both train and test set,
for example, if labels set contains k of A, 2k of B, 3k of C,
we get k' of A, 2k' of B, 3k' of C in training labels after splitting, and so is the test labels
"""
split = train_test_split(trainPaths,
                         trainLabels,
                         test_size=config.NUM_TEST_IMAGES,
                         stratify=trainLabels,
                         random_state=42)

(trainPaths, testPaths, trainLabels, testLabels) = split

# further split a portion of training set into validation set
split = train_test_split(
    trainPaths,
    trainLabels,
    test_size=config.NUM_VAL_IMAGES,
    stratify=trainLabels,
    random_state=42)

(trainPaths, valPaths, trainLabels, valLabels) = split

#############################
### save images into hdf5 ###
#############################

datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)
]

aspectAwarePreprocessor = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter(
        dims=(len(paths), 256, 256, 3), outputPath=outputPath)

    pbar = generate_progressbar(title="Building Dataset:", maxval=len(paths))

    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aspectAwarePreprocessor.preprocess(image)

        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # writer.add method intrinsically use "extend", which concatenate list
        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()

####################################################
### save means for mean subtraction normlization ###
####################################################

print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
fs = open(config.DATASET_MEAN_JSON, "w")
fs.write(json.dumps(D))
fs.close()
