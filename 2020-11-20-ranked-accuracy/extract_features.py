from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths

import numpy as np
import progressbar
import argparse
import random
import os

# extract flattened features

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
                help="path to output HDF5 file")
ap.add_argument("-b", "--batch-size", type=int, default=32,
                help="batch size of images to b e passed through netework")
ap.add_argument("-s", "--buffer-size", type=int, default=1000,
                help="size of feature extraction buffer")
args = vars(ap.parse_args())

batchSize = args["batch_size"]
imagePaths = list(paths.list_images(args["dataset"]))

# previously we use train_test_split from sklearn, but this is not feasible for dataset that is too large,
# so we shuffle the imagePaths instead
random.shuffle(imagePaths)

labels = [imgPath.split(os.path.sep)[-2] for imgPath in imagePaths]
labelEnconder = LabelEncoder()

# in fitting, labelEnconder will sort our array in alphabetical order,
# then transform our string into an integer in a canonical way
# note that our .labels in HDF5DatasetWriter instance is of int type.
labels = labelEnconder.fit_transform(labels)

# we use include_top=False to remove the top, i.e., the dense layer part.
model = VGG16(weights="imagenet", include_top=False)


dataset = HDF5DatasetWriter(
    (len(imagePaths), 512*7*7), args["output"], dataKey="features", bufferSize=args["buffer_size"])
dataset.storeClassLabels(labelEnconder.classes_)

widgets = ["Extracting Features: ",
           progressbar.Percentage(),
           " ",
           progressbar.Bar(),
           " ",
           progressbar.ETA()]

pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

for i in np.arange(0, len(imagePaths), batchSize):
    batchPaths = imagePaths[i:i+batchSize]
    batchLabels = labels[i:i+batchSize]
    batchImages = []

    for (j, imagePath) in enumerate(batchPaths):
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        # image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)

    # batchImages = np.vstack(batchImages)
    batchImages = np.array(batchImages)
    features = model.predict(batchImages, batch_size=batchSize)
    # flatten
    features = np.reshape(features, (-1, 512 * 7 * 7))
    dataset.add(features, batchLabels)
    pbar.update(i)

dataset.close()
pbar.finish()
