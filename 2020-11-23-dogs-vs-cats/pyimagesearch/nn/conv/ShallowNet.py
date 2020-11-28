from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, numOfClasses):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # the "same" padding ensure input and output have the same dimesion
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(numOfClasses))
        model.add(Activation("softmax"))

        return model
