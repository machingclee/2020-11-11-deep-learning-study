from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the image")
ap.add_argument("-m", "--model", type=str, default="vgg16",
                help="name of pre-trained network to uses")
args = vars(ap.parse_args())

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

if args["model"] not in MODELS.keys():
    raise AssertionError(
        "The --model command line argument should be a key in the MODELS dictionary")


# 3 kinds of preprocessing, detail: https://medium.com/@sci218mike/%E5%9C%96%E7%89%87%E9%A0%90%E8%99%95%E7%90%86%E4%BD%BF%E7%94%A8keras-applications-%E7%9A%84-preprocess-input-6ef0963a483e
# here it is either mean subtraction + normalization OR scaled by 127.5 followed by a subtraction by 1

inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input

print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

print("[INFO] loading and pre-processing image ...")

image = load_img(args["image"], target_size=inputShape)
print("[INFO] show the data type of image ...")
print(type(image))

image = img_to_array(image)
# image = np.expand_dims(image, axis=0)
# experiment, mimic expand_dims:
image = np.array([image])
image = preprocess(image)

print("[INFO] classifying image with '{}' ...".format(args["model"]))
predictions = model.predict(image)
P = imagenet_utils.decode_predictions(predictions)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}, {}: {:.2f}%".format(i+1, label, prob*100))

orig_img = cv2.imread(args["image"])
(imagenetId, label, prob) = P[0][0]

cv2.putText(
    orig_img,
    "Label: {}".format(label),
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

cv2.imshow("Classification", orig_img)
cv2.waitKey(0)
