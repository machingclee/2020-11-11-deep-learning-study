from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Input
from PIL import Image
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
from deeptools.utils import plot_model


def loadImage(imagePath, width=350):
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=width)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)

    return image


def deprocess(image):
    image = 255*(image+1.0)
    image = image/2.0
    image = tf.cast(image, tf.uint8)
    return image


def calculateLoss(image, model):
    image = tf.expand_dims(image, axis=0)
    layerActivations = model(image)
    print(layerActivations)

    losses = []
    for act in layerActivations:
        loss = tf.reduce_mean(act)
        loss = tf.reduce_mean(loss)
        losses.append(loss)

    return tf.reduce_sum(losses)


@tf.function
def deepDream(model, image, stepSize, eps=1e-8):
    with tf.GradientTape() as tape:
        # tensors being watched will be recorded and considered as a variable for later differentiation
        tape.watch(image)
        loss = calculateLoss(image, model)

    # derivative of loss w.r.t. image (and then evaluated at image)
    gradients = tape.gradient(loss, image)
    gradients = gradients/tf.math.reduce_std(gradients) + eps
    image = image + (gradients * stepSize)
    image = tf.clip_by_value(image, -1, 1)

    return (loss, image)


def runDeepDreamModel(model, image, iterations=100, stepSize=0.01):
    image = preprocess_input(image)
    for iteration in range(iterations):
        (loss, image) = deepDream(model, image, stepSize)

        if iteration % 25 == 0:
            print("[INFO] iteration{}, loss{}".format(iteration, loss))

    return deprocess(image)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-o", "--output", required=True, help="path to output dreamed image")
args = vars(ap.parse_args())

names = ["mixed3", "mixed5"]
OCTAVE_SCALE = 1.3
NUM_OCTAVES = 3

print("[INFO] loading image...")
originalImage = loadImage(args["image"])

print("[INFO] loading inception network...")
baseModel = InceptionV3(include_top=False, weights="imagenet")
baseModel.summary()

layers = [baseModel.get_layer(name).output for name in names]
dreamModel = tf.keras.Model(inputs=baseModel.input, outputs=layers)
image = tf.constant(originalImage)
baseShape = tf.cast(tf.shape(image)[:-1], tf.float32)

for n in range(NUM_OCTAVES):
    newShape = tf.cast(baseShape * (OCTAVE_SCALE ** n), tf.int32)
    image = tf.image.resize(image, newShape).numpy()
    image = runDeepDreamModel(model=dreamModel, image=image, iterations=200, stepSize=0.001)

finalImage = np.array(image)
Image.fromarray(finalImage).save(args["output"])
