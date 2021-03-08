import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


x = np.arange(0, 10, 0.05)
series = np.sin(x) + np.random.randn(x.shape[0])*0.1
plt.plot(x, series, label="sine with noise")

T = 10
X = []
Y = []

for t in range(0, len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y).reshape(-1, 1)


class TimeSeriesPredictionModel:
    @staticmethod
    def build(T):
        input = Input(shape=(T, 1))
        x = SimpleRNN(5)(input)
        print(tf.shape(x))
        x = Dense(1)(x)
        model = Model(input, x)
        return model


model: Model = TimeSeriesPredictionModel.build(T=T)
model.compile(loss="mse", optimizer=Adam(lr=0.1))


model.fit(X, Y, epochs=80)


Y = model.predict(X)
Y.shape

plt.plot(np.linspace(start=0, stop=10, num=190), Y.flatten(), label="prediction from Simple RNN")
plt.legend(loc='upper right')
plt.show()
