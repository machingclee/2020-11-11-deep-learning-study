from deeptools.io import HDF5DatasetWriter
from config import srcnn_config as config
from imutils import paths
from PIL import Image
import numpy as np
import shutil
import random
import PIL
import cv2
import os


for p in [config.IMAGES, config.TARGETS]:
    if not os.path.exists(p):
        os.makedirs(p)

print("[INFO] creating temporary images...")
imagePaths = list(paths.list_images(config.IMAGES_DATA_DIR))
random.shuffle(imagePaths)
total = 0

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    # make w and h a multiple of 2:
    w = w - int(w % config.SCALE)
    h = h - int(h % config.SCALE)
    image = image[0:h, 0:w]

    lowW = int(w * (1.0/config.SCALE))
    lowH = int(h * (1.0/config.SCALE))
    highW = int(lowW * (config.SCALE/1.0))
    highH = int(lowH * (config.SCALE/1.0))

    scaled = np.array(Image.fromarray(image).resize((lowW, lowH), resample=PIL.Image.BICUBIC))
    scaled = np.array(Image.fromarray(scaled).resize((highW, highH), resample=PIL.Image.BICUBIC))

    for y in range(0, h - config.INPUT_DIM + 1, config.STRIDE):
        for x in range(0, w - config.INPUT_DIM + 1, config.STRIDE):

            """
             (x, y)
                |
                v
                ###################
                ##               ##
                ##               ##
                ##    target     ##
                ##               ##
                ##               ##
                ###################
                |<-- INPUT_DIM -->|
                |<- STRIDE ->|
                  |< LABELSIZE >|
            """
            # crop out the learning image by sqaures, we overlap the next square
            crop = scaled[y:
                          y + config.INPUT_DIM,
                          x:
                          x + config.INPUT_DIM]

            # crop out padding area for the learning target:
            target = image[y + config.PAD:
                           y + config.PAD + config.LABEL_SIZE,
                           x + config.PAD:
                           x + config.PAD + config.LABEL_SIZE]

            cropPath = os.path.sep.join([config.IMAGES, "{}.png".format(total)])
            targetPath = os.path.sep.join([config.TARGETS, "{}.png".format(total)])

            cv2.imwrite(cropPath, crop)
            cv2.imwrite(targetPath, target)

            total = total + 1

print("[INFO] building HDF5 dataset...")
inputPaths = sorted(list(paths.list_images(config.IMAGES)))
targetPaths = sorted(list(paths.list_images(config.TARGETS)))

inputWriter = HDF5DatasetWriter(
    (len(inputPaths), config.INPUT_DIM, config.INPUT_DIM, 3),
    config.INPUTS_DB
)
targetWriter = HDF5DatasetWriter(
    (len(targetPaths), config.LABEL_SIZE, config.LABEL_SIZE, 3),
    config.TARGETS_DB
)

for (inputPath, targetPath) in zip(inputPaths, targetPaths):
    print("[INFO] writing {} and {}".format(inputPath, targetPath))
    inputImage = cv2.imread(inputPath)
    targetImage = cv2.imread(targetPath)
    inputWriter.add([inputImage], [-1])
    targetWriter.add([targetImage], [-1])

inputWriter.close()
targetWriter.close()

print("[INFO] cleaning up ...")
shutil.rmtree(config.IMAGES)
shutil.rmtree(config.TARGETS)
