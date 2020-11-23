from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
import numpy as np


class MiniVGGNet:
    @staticmethod
    def build(height, width, depth, numOfClasses):
        model = Sequential()
        inputShape = (height, width, depth)
        channelPositionIndex = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            channelPositionIndex = 1

        # 1st block
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        # from https://stackoverflow.com/questions/35784558/batch-normalization-axis-on-which-to-take-mean-and-variance?fbclid=IwAR1xVns3zMXJQxIXpgttcC-vNONMYxXYXc4FTQ9OeupASvBsIzu2NhJ8ryM

        # Batchnormalization is done across every feature, i.e., within each channel. In CNN we consider each matrix as a feature, for example, if we have an output of shape (n_H, n_W, 32) within hidden layer, then there are 32 features, we have 32 means and sd's to normalize every feature.
        model.add(BatchNormalization(axis=channelPositionIndex))

        # 2nd block
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelPositionIndex))

        # as stride is not provided, keras implicitly assumes our stride is of size (2, 2):
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 3rd block
        # again CONV => RELU => BN iterations
        # we increase the depth as the spatial dimension get decreased when the network goes deeper.
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelPositionIndex))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelPositionIndex))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4th block
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(numOfClasses))
        model.add(Activation("softmax"))

        return model
