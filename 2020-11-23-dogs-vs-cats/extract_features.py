from tensorflow.keras.applications import ResNet50
from pyimagesearch.utils import extract_features
from imutils import paths

import random
import os

trainingDataPath = "dataset/kaggle_dogs_vs_cats/train"
outputPath = "dataset/kaggle_dogs_vs_cats/hdf5/features2.hdf5"

# shuffle the image paths first and then get the corresponding label
# the way we extract labels depend on the folder structure,
# this time our label is attached to the file name, but sometimes dataset is classified already in different folders

imagePaths = list(paths.list_images(trainingDataPath))
random.shuffle(imagePaths)
labels = [path.split(os.path.sep)[-1].split(".")[0] for path in imagePaths]

# declare feature extraction model and the size of the batch of training data size.
# 100352 depends on the layer structure
featuresDim = (len(imagePaths), 100352)
model = ResNet50(weights="imagenet", include_top=False)

extract_features(featuresDim,
                 model,
                 imagePaths,
                 labels,
                 outputPath)
