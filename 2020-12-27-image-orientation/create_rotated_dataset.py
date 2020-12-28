from imutils import paths
from config import image_orientation_config as config
from deeptools.utils import generate_progressbar
import numpy as np
import argparse
import imutils
import random
import cv2
import os

imagePaths = list(paths.list_images(config.DATASET_DIR))[:10000]
random.shuffle(imagePaths)

pbar = generate_progressbar(maxval=len(imagePaths))
angles = {}

for (i, imagePath) in enumerate(imagePaths):
    angle = np.random.choice([0, 90, 180, 270])
    # numpy array:
    image = cv2.imread(imagePath)

    # skip images that fails to be loaded:
    if image is None:
        continue

    image = imutils.rotate_bound(image, angle)
    output_dir = os.path.sep.join([config.ROTATED_DATASET_DIR, str(angle)])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ext = imagePath[imagePath.rfind("."):]
    count = angles.get(angle, 0)
    output_path = os.path.sep.join([output_dir, "image_{}{}".format(str(count).zfill(5), ext)])

    cv2.imwrite(output_path, image)
    angles[angle] = count + 1
    pbar.update(i)

pbar.finish()

for angle in sorted(angles.keys()):
    print("[INFO] angle={}: {:,}".format(angle, angles[angle]))
