from imutils import paths
import numpy as np
import os
import shutil

TRAINS = os.path.sep.join(["D:", "datasets",  "ILSVRC2012", "train"])


def get_label_from_path(_path):
    return _path.split(os.path.sep)[-1].split("_")[0]


train_paths = paths.list_images(TRAINS)

for train_path in train_paths:
    classification = get_label_from_path(train_path)
    new_label_dir = os.path.sep.join([TRAINS, classification])
    imageName = train_path.split(os.path.sep)[-1]

    if not os.path.exists(new_label_dir):
        os.makedirs(new_label_dir)

    print("[INFO] moving {}...".format(train_path))
    shutil.move(train_path, os.path.join(new_label_dir, imageName))
