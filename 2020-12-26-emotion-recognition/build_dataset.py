from config import emotion_config as config
from deeptools.io import HDF5DatasetWriter
from deeptools.utils import generate_progressbar
import numpy as np

fs = open(config.INPUT_PATH)
fs.__next__()
(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

for row in fs:
    (label, image, usage) = row.strip().split(",")
    label = int(label)

    if config.NUM_CLASSES == 6:
        if label == 1:
            label = 0
        if label > 0:
            label = label-1

    image = np.array(image.split(" "), dtype="uint8")
    image = image.reshape((48, 48))

    if usage == "Training":
        trainImages.append(image)
        trainLabels.append(label)

    elif usage == "PrivateTest":
        valImages.append(image)
        valLabels.append(label)

    else:
        testImages.append(image)
        testLabels.append(label)

datasets = [
    (trainImages, trainLabels, config.TRAIN_HDF5),
    (valImages, valLabels, config.VAL_HDF5),
    (testImages, testLabels,  config.TEST_HDF5)
]

for (images, labels, hdf5_filePath) in datasets:
    print("[INFO] build {}".format(hdf5_filePath))
    n_images = len(images)
    writer = HDF5DatasetWriter((n_images, 48, 48), hdf5_filePath)

    for (i, (image, label)) in enumerate(zip(images, labels)):
        writer.add([image], [label])

    writer.close()
