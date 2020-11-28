from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras import backend as K


class LeNet:
    @staticmethod
    def build(height, width, depth, numOfClasses):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)

        """
        if padding = "same", after striding, if the number of columns of the submatrix is not 5,
        it will add 0 padding on the right until the convolution makes sense.
        """

        model.add(Conv2D(20, (5, 5),
                         padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(numOfClasses))
        model.add(Activation("softmax"))

        return model
