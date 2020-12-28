from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import shape
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers.pooling import MaxPooling2D


class ResNet:
    @staticmethod
    def residual_module(k_depth, strides, chan_dim, reduce_dim=False, reg=0.0001, bn_eps=2e-5, bn_mom=0.9):
        """
        In residual module, filter size is either (1, 1) or (3, 3).
        When (1, 1) is used, we learn local feature or control volume size.
        When (3, 3) is used, we learn the features and reduce spatial dimension at the same time.
        """
        def module(x):
            shortcut = x

            # k_depth will be chosen to be the depth of x, so that addition makes sense.

            bn_1 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(x)
            act_1 = Activation("relu")(bn_1)
            conv_1 = Conv2D(int(k_depth * 0.25), (1, 1), strides=(1, 1), use_bias=False, kernel_regularizer=l2(reg))(act_1)

            # reduce spatial dimension in this block, if needed

            bn_2 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(conv_1)
            act_2 = Activation("relu")(bn_2)
            conv_2 = Conv2D(int(k_depth * 0.25), (3, 3), strides=strides, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act_2)

            # retrieve depth in this block

            bn_3 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(conv_2)
            act_3 = Activation("relu")(bn_3)
            conv_3 = Conv2D(k_depth, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act_3)

            # when strides=(2, 2), we will also set reduce_dim = true, otherwise addition never make sense

            if reduce_dim:
                shortcut = Conv2D(k_depth, (1, 1), strides=strides, use_bias=False, kernel_regularizer=l2(reg))(act_1)

            x = add([conv_3, shortcut])

            return x

        return module

    @staticmethod
    def build(width, height, depth, n_classes, n_residuals_list, filters_depth, reg=0.0001, bn_eps=2e-5, bn_mom=0.9, dataset="cifar"):

        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        input = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bn_eps, momentum=bn_mom)(input)

        # Control the size before stacking residue modules:
        # First block does not have any residual module, we set n_residuals_list[0] = 0

        if dataset == "cifar":
            # cifar is small, we use a stride of (1, 1) to keep the spatial dimension

            x = Conv2D(filters_depth[0], (3, 3), strides=(1, 1), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)

        elif dataset == "tiny_imagenet":
            x = Conv2D(filters_depth[0], (5, 5), strides=(1, 1), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
            x = BatchNormalization(axis=chanDim, epsilon=bn_eps, momentum=bn_mom)(x)
            x = Activation("relu")(x)
            x = ZeroPadding2D((1, 1))(x)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)
            # x = ResNet.residual_module(filters_depth[0], (2, 2), chanDim, reduce_dim=True, reg=reg)

        # n_residuals_list[1:], filters_depth[1:] take effect:
        # By design, we don't reduce the spatial dimension in the first residual block

        for i in range(1, len(n_residuals_list)):
            strides = None
            if i == 1:
                strides = (1, 1)
            else:
                strides = (2, 2)
            x = ResNet.residual_module(filters_depth[i], strides, chanDim, reduce_dim=True, bn_eps=bn_eps, bn_mom=bn_mom)(x)
            # repeat to learn residual part
            for _ in range(0, n_residuals_list[i] - 1):
                x = ResNet.residual_module(filters_depth[i], (1, 1), chanDim, bn_eps=bn_eps, bn_mom=bn_mom)(x)

        # thus, x is downsampled from x.shape to x.shape/2^(len(stages)-1) after the for loop

        x = BatchNormalization(axis=chanDim, epsilon=bn_eps, momentum=bn_mom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        x = Flatten()(x)
        x = Dense(n_classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        model = Model(input, x, name="resnet")

        return model
