from config import imagenet_alexnet_config as config
from sklearn.model_selection import train_test_split
from deeptools.utils import ImagenetHelper
from deeptools.utils import generate_progressbar
import numpy as np
import json
import cv2

print("[INFO] loading image paths...")
imagenetHelper = ImagenetHelper(config)

(trainPaths, trainLabels) = imagenetHelper.buildTrainingSet()
(valPaths, valLabels) = imagenetHelper.buildValidationSet()

split = train_test_split(
    trainPaths, trainLabels,
    test_size=config.NUM_TEST_IMAGES,
    stratify=trainLabels,
    random_state=42
)

(trainPaths, testPaths, trainLabels, testLabels) = split

datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_MX_LIST),
    ("val", valPaths, valLabels, config.VAL_MX_LIST),
    ("test", testPaths, testLabels, config.TEST_MX_LIST)
]

(R, G, B) = ([], [], [])


for (dType, paths, labels, outputPath) in datasets:
    fs = open(outputPath, "w")
    pbar = generate_progressbar(maxval=len(paths))

    for(i, (path, label)) in enumerate(zip(paths, labels)):
        row = "\t".join([str(i), str(label), path])
        fs.write("{}\n".format(row))

        if dType == "train":
            image = cv2.imread(path)
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        pbar.update(i)

    pbar.finish()
    fs.close()

print("[INFO] serializing means...")
D = {}
D["R"] = np.mean(R)
D["G"] = np.mean(G)
D["B"] = np.mean(B)

fs = open(config.DATASET_MEAN, "w")
fs.write(json.dumps(D))
fs.close()
