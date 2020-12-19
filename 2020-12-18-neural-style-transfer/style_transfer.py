from typing import ChainMap
from config import style_transfer_config as config
from deeptools.nn.conv.NeuralStyle import NeuralStyle
import tensorflow as tf
import os


def loadImage(imagePath):
    maxDim = 512
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    longDim = max(shape)
    scale = maxDim/longDim

    newShape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, newShape)
    image = image[tf.newaxis, :]

    return image


def trainOneStep(image, styleTargets, contentTargets, extractor, opt):
    styleWeight = config.styleWeight / len(config.styleLayers)
    contentWeight = config.contentWeight / len(config.contentLayers)

    with tf.GradientTape() as tape:
        tape.watch(image)
        outputs = extractor(image)
        loss = extractor.styleContentLoss(outputs, styleTargets, contentTargets, styleWeight, contentWeight)
        loss = loss + config.tvWeight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(extractor.clipPixels(image))


opt = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.99, epsilon=1e-1)
print("[INFO] loading content and style images...")
styleImage = loadImage(config.styleImage)
contentImage = loadImage(config.contentImage)


contentLayers = config.contentLayers
styleLayers = config.styleLayers

extractor = NeuralStyle(styleLayers, contentLayers)
styleTargets = extractor(styleImage)["style"]
contentTargets = extractor(contentImage)["content"]

print("[INFO] training the style transfer model...")
image = tf.Variable(contentImage)
step = 0

for epoch in range(config.epochs):
    for i in range(config.stepsPerEpoch):
        trainOneStep(image, styleTargets, contentTargets, extractor, opt)
        step += 1

    print("[INFO] training steps: {}".format(step))
    imagePath = os.path.sep.join([config.intermOutputs, "epoch-{}.png".format(epoch)])
    extractor.tensorToImage(image).save(imagePath)

extractor.tensorToImage(image).save(config.finalImage)
