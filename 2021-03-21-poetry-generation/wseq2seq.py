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
from tensorflow.python.distribute.distribute_coordinator import _get_num_workers
from tensorflow.python.keras.backend import dropout

BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256
NUM_SAMPLES = 10000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

input_texts = []
target_texts = []
target_texts_inputs = []


t = 0
for line in open("./jpn.txt", encoding="utf8"):
    t = t + 1
    if t > NUM_SAMPLES:
        break

    if "\t" not in line:
        continue

    values = line.split("\t")
    input_text = values[0]
    translation = values[1]

    target_text = translation + " <eos>"
    target_text_input = "<sos> " + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_text_input)


tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS, filters="")
tokenizer_inputs.fit_on_texts(input_texts)
input_seqs = tokenizer_inputs.texts_to_sequences(input_texts)

word2idx_inputs = tokenizer_inputs.word_index
max_len_input = max(len(s) for s in input_seqs)


tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters="")
# make sure to save both <sos> and <eos>
# this is to save both <sos> and <eos>, very inefficient though
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)

target_seqs = tokenizer_outputs.texts_to_sequences(target_texts)
target_inputs_seqs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)


word2idx_outputs = tokenizer_outputs.word_index

# keras use word_count+1 as the index of UNKNOWN, the key is "None" by default.
# note that keras also starts the index at 1
# 0 is the padding value for input
num_words_output = len(word2idx_outputs) + 1

# max length of outputs seq
max_len_target = max(len(s) for s in target_seqs)

# pad the seqs, input_seqs come from training data
# by default padding="pre", i.e., add zeros at the beginning
encoder_inputs = pad_sequences(
    input_seqs,
    maxlen=max_len_input,
)

print("encoder_data.shape", encoder_inputs.shape)
print("encoder_inputs[0]", encoder_inputs[0])

decoder_inputs = pad_sequences(
    target_inputs_seqs,
    maxlen=max_len_target,
    padding="post"
)

print("decoder_inputs.shape", decoder_inputs.shape)
print("decoder_inputs[0]", decoder_inputs[0])

# for LSTM training part input and output are of same length in sentences size
decoder_targets = pad_sequences(
    target_seqs,
    maxlen=max_len_target,
    padding="post"
)

print("Loading word vectors...")
word2vec = {}

with open(os.path.join("./glove.6B.{}d.txt".format(EMBEDDING_DIM))) as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.array(values[1:], dtype="flaot32")
        word2vec[word] = vec

print("filling pre-trained embeddings")

# include 0:
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

# for embedding matrix, we are just interested in words in our training set:
# word2idx_inputs comes from tokenizer of our training inputs
for word, index in word2idx_inputs.items():
    word_vec = word2vec.get(word)
    if word_vec is not None:
        embedding_matrix[index] = word_vec

embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=max_len_input
)

decoder_targets_one_hot = np.zeros(
    (len(input_texts), max_len_target, num_words_output),
    dtype="float32"
)

# e.g., decoder_targets_one_hot[0] is an array of binarized vectors, [e_1, e_4, ..., e_10], where e_k denotes the standard basis in R^num_words_output
for i, wordDigits in enumerate(decoder_targets):
    for t, digit in enumerate(wordDigits):
        decoder_targets_one_hot[i, t, digit] = 1

# build the model
# input is expected to be an array of integers
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LATENT_DIM, return_state=True, dropout=0.5)
encoder_outputs, h, c = encoder(x)


encoder_states = [h, c]


decoder_inputs_placeholder = Input(shape=(max_len_target,))

# it is a different langauge, we use different embedding
decoder_embedding = Embedding(num_words_output, LATENT_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=0.5)
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs_x,
    initial_state=encoder_states
)

decoder_dense = Dense(num_words_output, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["acc"]
)
r = model.fit(
    [encoder_inputs, decoder_inputs], decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)


plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

plt.plot(r.history["acc"], label="acc")
plt.plot(r.history["val_acc"], label="val_acc")
plt.legend()
plt.show()

# num_words = 3
# tk = Tokenizer(num_words=num_words+1, oov_token="UNK")
# texts = ["my name is far faraway asdasd", "my name is","your name is"]
# tk.fit_on_texts(texts)
# # see #8092 below why I do these two line
# tk.word_index = {e:i for e,i in tk.word_index.items() if i <= num_words} # <= because tokenizer is 1 indexed
# tk.word_index[tk.oov_token] = num_words + 1

# # output
# print(tk.word_index)
# wihtout oov_token, UNK is replaced by "None"
# {'name': 1, 'my': 3, 'is': 2, 'UNK': 4}
# print(tk.texts_to_sequences(texts))
# [[3, 1, 2, 4, 4, 4], [3, 1, 2], [4, 1, 2]]
