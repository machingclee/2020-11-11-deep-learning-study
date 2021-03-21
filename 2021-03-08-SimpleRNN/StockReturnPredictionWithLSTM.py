from tensorflow.keras.layers import Input, SimpleRNN, GRU, Dropout, LSTM, Dense, Flatten, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
df = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv')

series = df["close"].values.reshape(-1, 1)
scalar = StandardScaler()
scalar.fit(series[:len(series) // 2])
series = scalar.transform(series).flatten()


df["prevClose"] = df["close"].shift(1)
df["Return"] = (df["close"] - df["prevClose"])/df["prevClose"]
df["Return"].hist()


u = np.array([1, 2])
v = np.array([3, 4])
