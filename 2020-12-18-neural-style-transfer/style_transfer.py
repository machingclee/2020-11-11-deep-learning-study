from typing import ChainMap
from config import style_transfer_config as config
from deeptools.nn.conv.NeuralStyle import NeuralStyle
import tensorflow as tf
import os


def loadImage(imagePath):
    maxDim = 512
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_image(image, channel=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    longDim = max(shape)
    scale = maxDim/longDim

    newShape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, newShape)
    image = image[tf.newaxis, :]

    return image


@tf.function
def trainOneStep(image, styleTargets, contentTargets, extractor: NeuralStyle, opt):
    styleWeight = config.styleWeight / len(config.styleLayers)
    contentWeight = config.contentWeight / len(config.contentLayers)

    with tf.GradientTape() as tape:
        outputs = extractor(image)

        loss = extractor.styleContentLoss(outputs, styleTargets, contentTargets, styleWeight, contentWeight)
        loss = loss + config.tvWeight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(extractor.clipPixels(image))
