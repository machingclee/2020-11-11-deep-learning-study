import os
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam,  SGD

MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 2000
LATENT_DIM = 25


input_texts = []
target_texts = []
for line in open("./robert_frost.txt", encoding="utf-8"):
    line = line.rstrip()
    if not line:
        continue

    input_line = "<sos> " + line
    target_line = line + " <eos>"
    input_texts.append(input_line)
    target_texts.append(target_line)

all_lines = input_texts + target_texts

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters="")
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

max_sequence_length_from_data = max(len(s) for s in input_sequences)

word2idx = tokenizer.word_index
print("Found {} unique tokens.".format(len(word2idx)))
assert("<sos>" in word2idx)
assert("<eos>" in word2idx)


max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCE_LENGTH)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding="post")
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding="post")
print("Shape of data tensor:", input_sequences.shape)

print("Loading word vectors")
word2vec = {}
with open("./glove.6B.50d.txt", encoding="utf-8") as f:
    # it is a space-separated text file in the format:
    # word vec[0] vec[1] vec[2] ... for each line
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.array(values[1:], dtype="float32")
        word2vec[word] = vec

print("Found {} word vectors".format(len(word2vec)))

print("filling pre-trained embeddings...")

# word index starts from 1, so we + 1 below
# number_words = vocab size
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, index in word2idx.items():
    if index < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # if word is not found from pretrained glove data,
            # leave the embedding zeros

            embedding_matrix[index] = embedding_vector

# the following mimic the function of LabelBinearizer:
# *_position means the "index" of this element in the array,
# this is to distinguish from the word "index" as index already carries the meaning of
# word -> integer mapping

one_hot_targets = np.zeros((len(input_sequences), max_sequence_length, num_words))
for target_sentence_position, target_sequence in enumerate(target_sequences):
    for word_position, word_index in enumerate(target_sequence):
        if word_index > 0:
            one_hot_targets[target_sentence_position, word_position, word_index] = 1

# weight: word_index |-> word_vector
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix]
)

print("Building Model ...")
input_ = Input(shape=(max_sequence_length,))
initial_h = Input(shape=(LATENT_DIM,))
initial_c = Input(shape=(LATENT_DIM,))

# -------------- define the first model --------------:
x = embedding_layer(input_)
lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name="lstm")
x, _, _ = lstm(x, initial_state=[initial_h, initial_c])

# output should be of size (None, max_sequence_length, num_words)
dense = Dense(num_words, activation="softmax")
output = dense(x)

model = Model(
    [input_, initial_h, initial_c],
    output
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(lr=0.01),
    metrics=["acc"]
)

print("Training Model...")
z = np.zeros((len(input_sequences), LATENT_DIM))
record = model.fit(
    [input_sequences, z, z],
    one_hot_targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)

print("model weight:")
print(model.get_layer(name="lstm").get_weights()[0])


plt.plot(record.history["loss"], label="loss")
plt.plot(record.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

plt.plot(record.history["acc"], label="acc")
plt.plot(record.history["val_acc"], label="val_acc")
plt.legend()
plt.show()


# -------------- define the second model --------------:
# i.e., of shape (None, 1), if there is one sample to predict,
# we need to use (1, 1) dimensional np array.
input2 = Input(shape=(1,))  # (None, 1)
x = embedding_layer(input2)  # (None, 50)
x, h, c = lstm(x, initial_state=[initial_h, initial_c])
#x.shape (None, 1, LATTEN_DIM)
output2 = dense(x)  # (None, 1, num_words)
sampling_model = Model(
    [input2, initial_h, initial_c],
    [output2, h, c]
)
print("sampling_model weight:")
print(sampling_model.get_layer(name="lstm").get_weights()[0])

idx2word = {index: word for word, index in word2idx.items()}


def sample_line():
    np_input = np.array([[word2idx["<sos>"]]])
    h = np.zeros((1, LATENT_DIM))
    c = np.zeros((1, LATENT_DIM))

    eos = word2idx["<eos>"]

    output_sentence = []

    for _ in range(max_sequence_length):
        o, h, c = sampling_model.predict([np_input, h, c])

        probs = o[0, 0]  # shape: (num_words,)
        if np.argmax(probs) == 0:
            print("this is strange")
        probs[0] = 0
        probs = probs / np.sum(probs, axis=0)
        # pick a sample randomly with the predict as highest probability to avoid boring result
        idx = np.random.choice(len(probs), p=probs)
        if idx == eos:
            break

        output_sentence.append(
            idx2word.get(idx, "<WTF {}>".format(idx))
        )

        np_input[0, 0] = idx

    return " ".join(output_sentence)

# generate 4 lines:


while True:
    for _ in range(4):
        print(sample_line())

    ans = input("---generate another? [Y/n]---")
    if ans and ans[0].lower().startswith("n"):
        break
