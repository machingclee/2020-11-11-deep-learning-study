from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import activations
from tensorflow.keras import Sequential


class FCHeadNet_FromFlatten:
    @staticmethod
    def build(baseModel, inputDimension, numOfClasses):
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(inputDimension, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(numOfClasses, activation="softmax")(headModel)
        return headModel
