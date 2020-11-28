from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.regularizers import l2
from keras import backend as K

import numpy as np


class AlexNet:
    @staticmethod
    def build(width, height, depth, numOfClasses, reg=0.0002):
        model = Sequential()
        inputShape = (height, width, depth)
        channelIndex = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            channelIndex = 1

        # for conv layer, batch normalization is feature-wise, i.e., "channel-wise"

        ###################
        ### first block ###
        ###################

        model.add(Conv2D(96,
                         (11, 11),
                         strides=(4, 4),
                         input_shape=inputShape,
                         padding="same",
                         kernel_regularizer=l2(reg)
                         ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelIndex))

        model.add(MaxPool2D(pool_size=(3, 3),
                            strides=(2, 2)
                            ))
        model.add(Dropout(0.25))

        ####################
        ### second block ###
        ####################

        model.add(Conv2D(256,
                         (5, 5),
                         padding="same",
                         kernel_regularizer=l2(reg)
                         ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelIndex))

        model.add(MaxPool2D(pool_size=(3, 3),
                            strides=(2, 2)
                            ))
        model.add(Dropout(0.25))

        ###################
        ### third block ###
        ###################

        model.add(Conv2D(384,
                         (3, 3),
                         padding="same",
                         kernel_regularizer=l2(reg)
                         ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelIndex))

        model.add(Conv2D(384,
                         (3, 3),
                         padding="same",
                         kernel_regularizer=l2(reg)
                         ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelIndex))

        model.add(Conv2D(256,
                         (3, 3),
                         padding="same",
                         kernel_regularizer=l2(reg)
                         ))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelIndex))

        model.add(MaxPool2D(pool_size=(3, 3),
                            strides=(2, 2)
                            ))
        model.add(Dropout(0.25))

        ###################
        ### forth block ###
        ###################

        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        ###################
        ### fifth block ###
        ###################

        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(numOfClasses, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        return model
