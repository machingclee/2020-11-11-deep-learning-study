from tensorflow.keras import activations, losses, optimizers
from tensorflow.keras.layers import Reshape, Conv2D, MaxPool2D, Flatten, Dense, Input, LeakyReLU, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf

DIMEN = 128  # dimension of the image

input_shape = ((DIMEN**2) * 3, )
convolution_shape = (DIMEN, DIMEN, 3)
strides = 1

seq_models = [
    Reshape(input_shape=input_shape, target_shape=convolution_shape),
    Conv2D(32, (4, 4)),
    LeakyReLU(),
    Conv2D(32, (4, 4)),
    LeakyReLU(),
    MaxPool2D(pool_size=(3, 3), strides=1),
    Conv2D(64, kernel_size=(3, 3)),
    LeakyReLU(),
    Conv2D(64, kernel_size=(3, 3)),
    LeakyReLU(),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation=activations.sigmoid)
]

seq_model = tf.keras.Sequential(seq_models)

input_x1 = Input(shape=input_shape)
input_x2 = Input(shape=input_shape)

output_x1 = seq_model(input_x1)
output_x2 = seq_model(input_x2)

distance_euclid = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([output_x1, output_x2])
outputs = Dense(1, activation=activations.sigmoid)(distance_euclid)
model = Model([input_x1, input_x2], outputs)

model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.0001))
