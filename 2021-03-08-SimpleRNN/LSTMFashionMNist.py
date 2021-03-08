from tensorflow.keras.layers import Input, SimpleRNN, GRU, Dropout, LSTM, Dense, Flatten, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import shape
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.layers.core import Dropout
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

X = x_train
y_train.shape
lb = LabelBinarizer()
Y = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


class LSTM_MNist_Model:
    @staticmethod
    def build(T):
        input = Input(shape=(T, T))
        x = input
        x = LSTM(128)(x)
        x = Dense(64)(x)
        x = Dropout(0.4)(x)
        x = Dense(32)(x)
        x = Dropout(0.4)(x)
        x = Dense(10, activation="softmax")(x)

        return Model(input, x)


model = LSTM_MNist_Model.build(28)
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr=1e-2),
              metrics=["accuracy"]
              )

record = model.fit(X, Y,
                   epochs=10,
                   validation_data=(x_test, y_test))


# test the model:
i = np.random.choice(range(y_test.shape[0]), 1)[0]
plt.imshow(x_test[i])
feed = x_test[i][np.newaxis, ...]
print(np.argmax(model.predict(feed)))
