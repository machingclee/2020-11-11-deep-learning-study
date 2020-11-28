from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense


class FCHeadNet_FromFlatten:
    @staticmethod
    def build(baseModel, inputDimension, numOfClasses):
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(inputDimension, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(numOfClasses, activation="softmax")(headModel)
        return headModel
