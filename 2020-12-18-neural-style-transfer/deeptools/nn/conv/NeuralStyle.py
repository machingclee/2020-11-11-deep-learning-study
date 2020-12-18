from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image

import tensorflow as tf
import numpy as np


class NeuralStyle(Model):
    def __init__(self, styleLayers, contentLayers):
        super(NeuralStyle, self).__init__()

        self.vgg = self.vggLayers(styleLayers+contentLayers)

        self.styleLayers = styleLayers
        self.contentLayers = contentLayers
        self.n_styleLayers = len(styleLayers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessedInput = preprocess_input(inputs)
        outputs = self.vgg(preprocessedInput)
        (styleOutputs, contentOutputs) = (outputs[:self.n_styleLayers], outputs[self.n_styleLayers])

        styleOutputs = [self.gramMatrix(styleOutput) for styleOutput in styleOutputs]

        contentDict = {contentName: value for contentName, value in zip(self.contentLayers, contentOutputs)}
        styleDict = {styleName: value for styleName, value in zip(self.styleLayers, styleOutputs)}

        return {"content": contentDict, "style": styleDict}

    @staticmethod
    def vggLayers(layerNames):
        vgg = VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layerNames]
        model = Model([vgg.input], outputs)

        return model

    @staticmethod
    def gramMatrix(inputTensor):
        result = tf.linalg.einsum("bijc,bijd->bcd", inputTensor, inputTensor)
        inputShape = tf.shape(inputTensor)
        locations = tf.cast(inputShape[1]*inputShape[2], tf.float32)

        return (result/locations)
