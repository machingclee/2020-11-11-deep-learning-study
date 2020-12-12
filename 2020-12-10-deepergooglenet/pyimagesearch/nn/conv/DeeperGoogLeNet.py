from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


class DeeperGoogLeNet:
    @staticmethod
    def conv_module(k_depth, k_size, strides, chanDim, padding="same", reg=0.0005, name=None):

        def layer(x):
            (convName, bnName, actName) = (None, None, None)

            if name is not None:
                convName = name + "_conv"
                bnName = name + "_bn"
                actName = name + "_act"

            x = Conv2D(k_depth, k_size, strides=strides, padding=padding, kernel_regularizer=l2(reg), name=convName)(x)
            x = BatchNormalization(axis=chanDim, name=bnName)(x)
            x = Activation("relu", name=actName)(x)
            return x

        return layer

    @staticmethod
    def inception_module(
            depth1x1, depth3x3Reduce, depth3x3, depth5x5Reduce, depth5x5, depth1x1Proj, chanDim, stage, reg=0.0005):
        """
        depthNxNReduce mean: first reduce the depth down to depthNxNReduce and then apply NxN conv layer.
        """

        def layer(x):
            x_1 = DeeperGoogLeNet.conv_module(depth1x1, (1, 1), (1, 1), chanDim, reg=reg, name=stage+"_first")(x)

            x_2 = DeeperGoogLeNet.conv_module(depth3x3Reduce, (1, 1), (1, 1), chanDim, reg=reg, name=stage+"_second_1")(x)
            x_2 = DeeperGoogLeNet.conv_module(depth3x3, (3, 3), (1, 1), chanDim, reg=reg, name=stage+"_second_2")(x_2)

            x_3 = DeeperGoogLeNet.conv_module(depth5x5Reduce, (1, 1), (1, 1), chanDim, reg=reg, name=stage+"third_1")(x)
            x_3 = DeeperGoogLeNet.conv_module(depth5x5, (5, 5), (1, 1), chanDim, reg=reg, name=stage+"_third_2")(x_3)

            x_4 = MaxPool2D((3, 3), strides=(1, 1), padding="same", name=stage+"_pool")(x)
            x_4 = DeeperGoogLeNet.conv_module(depth1x1Proj, (1, 1), (1, 1), chanDim,
                                              reg=reg, name=stage+"_fourth_1")(x_4)

            x = concatenate([x_1, x_2, x_3, x_4], axis=chanDim, name=stage+"_mixed")
            return x

        return layer

    @staticmethod
    def build(width, height, depth, n_classes, reg=0.0005):
        inputShape = (width, height, depth)
        chanDim = -1

        if K.image_data_format == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        input = Input(shape=inputShape)
        x = DeeperGoogLeNet.conv_module(64, (5, 5), (1, 1), chanDim, reg=reg, name="block1")(input)
        x = MaxPool2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)
        x = DeeperGoogLeNet.conv_module(64, (1, 1), (1, 1), chanDim, reg=reg, name="block2")(x)
        x = DeeperGoogLeNet.conv_module(192, (3, 3), (1, 1), chanDim, reg=reg, name="block3")(x)
        x = MaxPool2D((3, 3), strides=(2, 2), padding="same", name="pool2")(x)

        # as the spatial dimension is reduced, we can be bold to learn it with more filters (higher depth)
        # when we go deeper, more and more filters are needed to learn the features from the previous layer
        # this is also to compensate the reduction in spatial dimension when we do pooling
        x = DeeperGoogLeNet.inception_module(64, 96, 128, 16, 32, 32, chanDim, "3a", reg=reg)(x)
        x = DeeperGoogLeNet.inception_module(128, 128, 192, 32, 96, 64, chanDim, "3b", reg=reg)(x)
        x = MaxPool2D((3, 3), strides=(2, 2), padding="same", name="pool3")(x)

        x = DeeperGoogLeNet.inception_module(192, 96, 208, 16, 48, 64, chanDim, "4a", reg=reg)(x)
        x = DeeperGoogLeNet.inception_module(160, 112, 224, 24, 64, 64, chanDim, "4b", reg=reg)(x)
        x = DeeperGoogLeNet.inception_module(128, 128, 256, 24, 64, 64, chanDim, "4c", reg=reg)(x)
        x = DeeperGoogLeNet.inception_module(112, 144, 288, 32, 64, 64, chanDim, "4d", reg=reg)(x)
        x = DeeperGoogLeNet.inception_module(256, 160, 320, 32, 128, 128, chanDim, "4e", reg=reg)(x)
        x = MaxPool2D((3, 3), strides=(2, 2), padding="same", name="pool4")(x)

        x = AveragePooling2D((4, 4), name="pool5")(x)
        x = Dropout(0.4, name="dropout")(x)

        x = Flatten(name="flatten")(x)
        x = Dense(n_classes, kernel_regularizer=l2(reg), name="labels")(x)
        x = Activation("softmax", name="softmax")(x)

        model = Model(input, x, name="googlenet")

        return model
