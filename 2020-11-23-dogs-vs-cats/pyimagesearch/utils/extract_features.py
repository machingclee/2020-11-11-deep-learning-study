# usage: python extract_features.py --dataset dataset/kaggle_dogs_vs_cats/train --output dataset/kaggle_dogs_vs_cats/hdf5/features.hdf5

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from pyimagesearch.utils import generate_progressbar

import numpy as np
import random


def extract_features(featuresDim,
                     model,
                     imagePaths=[],
                     labels=[],
                     outputPath="",
                     batchSize=16,
                     bufferSize=1000):
    """
    featuresDim: (-1, m), where m is the flattened output length for each row.

    labels: a list of string, the label, it will be encoded and stored inside the dataset as a list of integers
    """
    print("[INFO] loading images ...")
    random.shuffle(imagePaths)

    labelEncoder = LabelEncoder()
    # this is not a binarizer, i.e., the fit_transform return a list of integers, but not a list of list of integers
    labels = labelEncoder.fit_transform(labels)
    dataset = HDF5DatasetWriter(dims=featuresDim,
                                outputPath=outputPath,
                                dataKey="features",
                                bufferSize=bufferSize)
    dataset.storeClassLabels(labelEncoder.classes_)

    pbar = generate_progressbar(maxval=len(imagePaths))

    for i in np.arange(0, len(imagePaths), batchSize):
        batchPaths = imagePaths[i:i+batchSize]
        batchLabels = labels[i:i+batchSize]
        batchImages = []

        for (j, imagePath) in enumerate(batchPaths):
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)

            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)

            batchImages.append(image)

        batchImages = np.vstack(batchImages)
        features = model.predict(batchImages, batch_size=batchSize)

        features = features.reshape((-1, featuresDim[1]))

        dataset.add(features, batchLabels)
        pbar.update(i)

    dataset.close()
    pbar.finish()
