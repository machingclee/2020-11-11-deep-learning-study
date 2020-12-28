from sys import path
from config import image_orientation_config as config
from numpy.core.numeric import require
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths

import os
import numpy as np
import argparse
import pickle
import imutils
import h5py
import cv2

FEATURE_HDF5 = config.FEATURES_HDF5
ROTATED_DIR = config.ROTATED_DATASET_DIR
TO_BE_ADJUST_DIR = os.path.sep.join([config.PRODJECT_DIR, "test-rotation-sample"])

db = h5py.File(FEATURE_HDF5)
labelNames = [int(angle) for angle in db["label_names"]]
db.close()

print("[INFO] sampling images...")
imagePaths = paths.list_images(TO_BE_ADJUST_DIR)
# imagePaths = list(paths.list_images(ROTATED_DIR))
# imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

vgg = VGG16(weights="imagenet", include_top=False)
model = load_model(os.path.sep.join([config.PRODJECT_DIR, "output", "orientation-15.hdf5"]))

for imagePath in imagePaths:
    orig = cv2.imread(imagePath)

    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    features = vgg.predict(image)
    features = features.reshape((-1, 512*7*7))

    angle = model.predict(features)
    print("[INFO] output of our logistic model: ")
    print(angle)
    angle = labelNames[angle.argmax()]

    rotated = imutils.rotate_bound(orig, 360 - int(angle))
    cv2.imshow("original", orig)
    cv2.imshow("corrected", rotated)

    cv2.waitKey(0)
