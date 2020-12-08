from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report
import numpy as np
import os

MODEL_PATH = os.path.sep.join(["models", "minigooglenet_cifar10.hdf5"])

((trainX, _), (testX, testY)) = cifar10.load_data()
testX = testX.astype("float32")
# recall that we have done mean subtraction on training,
# to be fair we have to do the same for test set, subtracting the same quantity
mean = np.mean(trainX, axis=0)
testX = testX - mean

labels = ["airplanes", "cars", "birds", "cats", "deer",
          "dogs", "frogs", "horses", "ships", "and trucks"]

minigooglenet = load_model(MODEL_PATH)
pred = minigooglenet.predict(testX)
report = classification_report(pred.argmax(axis=1),
                               testY,
                               target_names=labels)
print(report)
