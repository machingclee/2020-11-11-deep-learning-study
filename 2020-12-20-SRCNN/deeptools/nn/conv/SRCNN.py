from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


class SRCNN:
    @staticmethod
    def build(width, height, depth):
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # he_normal works well with relu activation
        input = Input(shape=inputShape)
        x = Conv2D(64, (9, 9), kernel_initializer="he_normal")(input)
        x = Activation("relu")(x)
        x = Conv2D(32, (1, 1), kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)
        x = Conv2D(depth, (5, 5), kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)

        return Model(input, x)
