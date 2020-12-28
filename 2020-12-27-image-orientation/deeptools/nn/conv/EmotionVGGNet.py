from operator import mod
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K


class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, n_classes):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        input = Input(shape=inputShape)

        x = input
        # first block
        for _ in range(0, 2):
            x = Conv2D(32, (3, 3),  padding="same", kernel_initializer="he_normal")(x)
            x = ELU()(x)
            x = BatchNormalization(axis=chanDim)(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # second block
        for _ in range(0, 2):
            x = Conv2D(64, (3, 3), kernel_initializer="he_normal", padding="same")(x)
            x = ELU()(x)
            x = BatchNormalization(axis=chanDim)(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # third block
        x = Flatten()(x)
        for _ in range(0, 2):
            x = Dense(64, kernel_initializer="he_normal")(x)
            x = ELU()(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)

        x = Dense(n_classes, kernel_initializer="he_normal")(x)
        x = Activation("softmax")(x)

        return Model(input, x)
