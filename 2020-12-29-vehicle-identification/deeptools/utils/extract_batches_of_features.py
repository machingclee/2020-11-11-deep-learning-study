# purpose: save batches of features and batches of labels


from deeptools.utils.generate_progressbar import generate_progressbar
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from deeptools.io import HDF5DatasetWriter
from imutils import paths

import numpy as np
import random
import os


model = VGG16(weights="imagenet", include_top=False)


def extract_batches_of_features(dataset_dir, output_path, batch_size=64, buffer_size=1000, model=model):
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset_dir))
    random.shuffle(imagePaths)

    labels = [_path.split(os.path.sep)[-2] for _path in imagePaths]
    labelEncoder = LabelEncoder()
    labels = labelEncoder.fit_transform(labels)

    dataset = HDF5DatasetWriter(
        dims=(len(imagePaths), 512*7*7),
        outputPath=output_path,
        dataKey="features",
        bufferSize=buffer_size
    )

    dataset.storeClassLabels(labelEncoder.classes_)

    pbar = generate_progressbar(maxval=len(imagePaths))

    for i in np.arange(0, len(imagePaths), batch_size):
        batchPaths = imagePaths[i:i+batch_size]
        batchLabels = labels[i:i+batch_size]
        batchImages = []

        for (j, imagePath) in enumerate(batchPaths):
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            # mean subtraction from imagenet dataset.
            image = imagenet_utils.preprocess_input(image)

            batchImages.append(image)

        batchImages = np.vstack(batchImages)
        features = model.predict(batchImages, batch_size=batch_size)
        features = features.reshape((-1, 512*7*7))

        dataset.add(features, batchLabels)
        pbar.update(i)

    dataset.close()
    pbar.finish()
