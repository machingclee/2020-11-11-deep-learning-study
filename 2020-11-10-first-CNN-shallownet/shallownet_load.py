from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import ResizePreprocessor
from pyimagesearch.datasets import DatasetLoader
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# constants
inputImageWidth = 32
inputImageHeight = 32

# arguments from initilization script
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

# load data and make predictions
classLabels = ["cat", "dog", "panda"]

imagePaths = np.array(list(paths.list_images(args["dataset"])))
subsetIndexs = np.random.randint(0, len(imagePaths), size=(10,))
subImagePaths = imagePaths[subsetIndexs]

resizePreprocessor = ResizePreprocessor(
    width=inputImageWidth,    height=inputImageHeight
)
imageToArrayPreprocessor = ImageToArrayPreprocessor()

preprocessors = [
    resizePreprocessor,
    imageToArrayPreprocessor
]

datasetLoader = DatasetLoader(preprocessors=preprocessors)
(data, labels) = datasetLoader.load(subImagePaths)
data = data.astype("float")/255.0

model = load_model(args["model"])
predictions = model.predict(data, batch_size=10).argmax(axis=1)

for (i, imgPath) in enumerate(subImagePaths):
    image = cv2.imread(imgPath)
    cv2.putText(
        image,
        "Label: {}".format(classLabels[predictions[i]]),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    cv2.imshow("Image", image)
    cv2.waitKey(0)
