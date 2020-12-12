from pyimagesearch.io import HDF5DatasetWriter
from pyimagesearch.utils import generate_progressbar
import cv2
import json
import numpy as np
import os


def dataset_to_hdf5(datasetType, imagePaths,
                    binarizedLabels, outputPath, trainingMeanJsonPath=None):
    """
    if datasetType is "train", it will produce a mean file to trainingMeanJsonPath (if exists)
    """

    writer = HDF5DatasetWriter((len(binarizedLabels), 64, 64, 3), outputPath)
    pbar = generate_progressbar(maxval=len(binarizedLabels))
    (R, G, B) = ([], [], [])

    for (i, (imagePath, label)) in enumerate(zip(imagePaths, binarizedLabels)):
        image = cv2.imread(imagePath)

        if datasetType == "train" and trainingMeanJsonPath is not None:
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image], [label])
        pbar.update(i)

    if datasetType == "train" and trainingMeanJsonPath is not None:
        fs = open(trainingMeanJsonPath, "w+")
        fs.write(json.dumps(
            {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
        ))
        fs.close()

    pbar.finish()
    writer.close()
