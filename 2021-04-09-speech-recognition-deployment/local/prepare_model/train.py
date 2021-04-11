import json
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import activations, regularizers, optimizers, losses
from sklearn.model_selection import train_test_split


DATA_PATH = "./data.json"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
SAVE_MODEL_PATH = "model.hdf5"
NUM_KEYWORDS = 10


def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y


def get_data_splits(data_path, test_size=0.1, val_size=0.1):
    # load dataset
    X, y = load_dataset(data_path)
    print("X.shape, y.shape", X.shape, y.shape)

    # create train/valiation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # convert inputs from 2d to 3d arrays
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size
    )
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape, learning_rate):
    input = Input(input_shape)

    x = Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(l2=0.001))(input)
    x = Activation(activations.relu)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(l2=0.001))(x)
    x = Activation(activations.relu)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(32, (2, 2), kernel_regularizer=regularizers.l2(l2=0.001))(x)
    x = Activation(activations.relu)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation(activation=activations.relu)(x)
    x = Dropout(0.3)(x)
    x = Dense(NUM_KEYWORDS)(x)

    output = Activation(activation=activations.softmax)(x)

    model = Model(input, output)
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    # use sparse_categorical_crossentropy because our target labels are just integers, not an one-hotted output
    model.compile(
        optimizer=optimizer,
        loss=losses.sparse_categorical_crossentropy,
        metrics=["acc"]
    )

    model.summary()

    return model


def main():
    # load train/validation/test data splits
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(DATA_PATH)

    # build the CNN model
    # (#segments, #coefficients, 1)
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape, LEARNING_RATE)

    # train the model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    # evaluate the model
    test_error, test_acc = model.evaluate(X_test, y_test)
    print(f"Test error:{test_error}, test accuracy: {test_acc}")

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    main()


"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 44, 13, 1)]       0
_________________________________________________________________
conv2d (Conv2D)              (None, 42, 11, 64)        640       
_________________________________________________________________
activation (Activation)      (None, 42, 11, 64)        0
_________________________________________________________________
batch_normalization (BatchNo (None, 42, 11, 64)        256       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 21, 6, 64)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 19, 4, 32)         18464     
_________________________________________________________________
activation_1 (Activation)    (None, 19, 4, 32)         0
_________________________________________________________________
batch_normalization_1 (Batch (None, 19, 4, 32)         128       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 10, 2, 32)         0
_________________________________________________________________

flatten (Flatten)            (None, 160)               0
_________________________________________________________________
dense (Dense)                (None, 64)                10304
_________________________________________________________________
activation_3 (Activation)    (None, 64)                0
_________________________________________________________________
dropout (Dropout)            (None, 64)                0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650
_________________________________________________________________
activation_4 (Activation)    (None, 10)                0
=================================================================
Total params: 34,698
Trainable params: 34,442
Non-trainable params: 256
_________________________________________________________________
"""
