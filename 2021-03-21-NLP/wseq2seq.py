import os
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


class WSeq2Seq:
    BATCH_SIZE = 64
    EPOCHS = 100
    LATENT_DIM = 256
    NUM_SAMPLES = 10000
    MAX_SEQUENCE_LENGTH = 100
    MAX_VOCAB_SIZE = 20000
    EMBEDDING_DIM = 100

    encoder_texts_input = []
    decoder_texts_output = []
    decoder_texts_input = []

    encoder_input_seq = []
    decoder_output_seq = []
    decoder_input_seq = []

    encoder_input_tokenizer: Tokenizer = None
    decoder_input_tokenizer: Tokenizer = None

    encoder_input_word_to_index = None
    decoder_input_word_to_index = None
    index_to_word_eng = None
    index_to_word_trans = None

    max_encoder_input_seq_length = 0
    max_decoder_input_seq_length = 0
    encoder_input_vocab_size = 0
    decoder_input_vocab_size = 0

    padded_encoder_inputs = []
    padded_decoder_inputs = []
    padded_decoder_outouts = []

    encoder_word_embedding_layer: Embedding = None
    decoder_one_hot_target = None

    encoder_model: Model = None
    decoder_model: Model = None

    @staticmethod
    def initialize():
        WSeq2Seq.generate_text_array()
        WSeq2Seq.generate_encoder_input_tokenizer()
        WSeq2Seq.generate_decoder_input_tokenizer()
        WSeq2Seq.generate_encoder_input_word_to_index()
        WSeq2Seq.generate_decoder_input_word_to_index()
        WSeq2Seq.transform_texts_to_seqs()
        WSeq2Seq.generate_input_vocab_sizes()
        WSeq2Seq.generate_input_max_seq_lengths()
        WSeq2Seq.generate_padded_seqs()
        WSeq2Seq.generate_encoder_word_embedding_layer()
        WSeq2Seq.generate_decoder_one_hot_target()
        WSeq2Seq.generate_index_to_words()

    @staticmethod
    def generate_text_array():
        t = 0
        for line in open("./jpn.txt", encoding="utf8"):
            t = t + 1
            if t > WSeq2Seq.NUM_SAMPLES:
                break

            if "\t" not in line:
                continue

            values = line.split("\t")
            input_text = values[0]
            translation = values[1]

            target_text = translation + " <eos>"
            target_text_input = "<sos> " + translation

            WSeq2Seq.encoder_texts_input.append(input_text)
            WSeq2Seq.decoder_texts_output.append(target_text)
            WSeq2Seq.decoder_texts_input.append(target_text_input)

    @staticmethod
    def generate_encoder_input_tokenizer():
        WSeq2Seq.encoder_input_tokenizer = Tokenizer(num_words=WSeq2Seq.MAX_VOCAB_SIZE, filters="")
        WSeq2Seq.encoder_input_tokenizer.fit_on_texts(WSeq2Seq.encoder_texts_input)

    @staticmethod
    def generate_decoder_input_tokenizer():
        WSeq2Seq.decoder_input_tokenizer = Tokenizer(num_words=WSeq2Seq.MAX_VOCAB_SIZE, filters="")
        # just to make sure both <sos> and <tos> are included, not efficient
        WSeq2Seq.decoder_input_tokenizer.fit_on_texts(WSeq2Seq.decoder_texts_output + WSeq2Seq.decoder_texts_input)

    @staticmethod
    def generate_encoder_input_word_to_index():
        WSeq2Seq.encoder_input_word_to_index = WSeq2Seq.encoder_input_tokenizer.word_index

    @staticmethod
    def generate_decoder_input_word_to_index():
        WSeq2Seq.decoder_input_word_to_index = WSeq2Seq.decoder_input_tokenizer.word_index

    @staticmethod
    def transform_texts_to_seqs():
        WSeq2Seq.encoder_input_seq = WSeq2Seq.encoder_input_tokenizer.texts_to_sequences(WSeq2Seq.encoder_texts_input)
        WSeq2Seq.decoder_output_seq = WSeq2Seq.decoder_input_tokenizer.texts_to_sequences(WSeq2Seq.decoder_texts_output)
        WSeq2Seq.decoder_input_seq = WSeq2Seq.decoder_input_tokenizer.texts_to_sequences(WSeq2Seq.decoder_texts_input)

    @staticmethod
    def generate_input_vocab_sizes():
        # +1 because we also include padding index 0 which correspond to nth in our vocab list
        # this becomes ncessary in creating embedding matrix
        # note that keras use word_count + 1 as the index of UNKNOWN, the key is "None" by default (already generated in word_index)
        WSeq2Seq.encoder_input_vocab_size = len(WSeq2Seq.encoder_input_tokenizer.word_index) + 1
        WSeq2Seq.decoder_input_vocab_size = len(WSeq2Seq.decoder_input_tokenizer.word_index) + 1

    @staticmethod
    def generate_input_max_seq_lengths():
        WSeq2Seq.max_encoder_input_seq_length = max(len(s) for s in WSeq2Seq.encoder_input_seq)
        WSeq2Seq.max_decoder_input_seq_length = max(len(s) for s in WSeq2Seq.decoder_input_seq)

    @staticmethod
    def generate_padded_seqs():
        WSeq2Seq.padded_encoder_inputs = pad_sequences(
            WSeq2Seq.encoder_input_seq,
            maxlen=WSeq2Seq.max_encoder_input_seq_length
        )
        # for LSTM training part input and output are of same length in sentences size
        WSeq2Seq.padded_decoder_inputs = pad_sequences(
            WSeq2Seq.decoder_input_seq,
            maxlen=WSeq2Seq.max_decoder_input_seq_length,
            padding="post"
        )
        WSeq2Seq.padded_decoder_outouts = pad_sequences(
            WSeq2Seq.decoder_output_seq,
            maxlen=WSeq2Seq.max_decoder_input_seq_length,
            padding="post"
        )

    @staticmethod
    def generate_encoder_word_embedding_layer():
        print("Loading word vectors...")
        word_to_vec = {}
        with open(os.path.join("./glove.6B.{}d.txt".format(WSeq2Seq.EMBEDDING_DIM)), encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.array(values[1:], dtype="float32")
                word_to_vec[word] = vec

        print("filling pre-trained embeddings")
        vocab_size = min(
            WSeq2Seq.MAX_VOCAB_SIZE,
            len(WSeq2Seq.encoder_input_word_to_index) + 1
        )
        embedding_matrix = np.zeros((vocab_size, WSeq2Seq.EMBEDDING_DIM))

        # for embedding matrix, we are just interested in words in our training set:
        for word, index in WSeq2Seq.encoder_input_word_to_index.items():
            word_vec = word_to_vec.get(word)
            if word_vec is not None:
                embedding_matrix[index] = word_vec

        WSeq2Seq.encoder_word_embedding_layer = Embedding(
            vocab_size,
            WSeq2Seq.EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=WSeq2Seq.max_encoder_input_seq_length
        )

    @staticmethod
    def generate_decoder_one_hot_target():
        decoder_one_hot_targets = np.zeros(
            (
                len(WSeq2Seq.decoder_texts_output),
                WSeq2Seq.max_decoder_input_seq_length,
                WSeq2Seq.decoder_input_vocab_size
            ),
            dtype="float32"
        )
        # e.g., decoder_targets_one_hot[0] is an array of binarized vectors, [e_1, e_4, ..., e_10], where e_k denotes the standard basis in R^num_words_output
        for i, wordDigits in enumerate(WSeq2Seq.decoder_output_seq):
            for t, digit in enumerate(wordDigits):
                decoder_one_hot_targets[i, t, digit] = 1

        WSeq2Seq.decoder_one_hot_target = decoder_one_hot_targets

    @staticmethod
    def generate_index_to_words():
        WSeq2Seq.index_to_word_eng = {v: k for k, v in WSeq2Seq.encoder_input_word_to_index.items()}
        WSeq2Seq.index_to_word_trans = {v: k for k, v in WSeq2Seq.decoder_input_word_to_index.items()}

    @staticmethod
    def build_encoder_decoder_model():
        # build the model
        # input is expected to be a batch of arrays of integers
        encoder_inputs_placeholder = Input(shape=(WSeq2Seq.max_encoder_input_seq_length,))
        x = WSeq2Seq.encoder_word_embedding_layer(encoder_inputs_placeholder)
        encoder = LSTM(WSeq2Seq.LATENT_DIM, return_state=True, dropout=0.5)
        _, h, c = encoder(x)
        encoder_states = [h, c]

        decoder_inputs_placeholder = Input(shape=(WSeq2Seq.max_decoder_input_seq_length,))
        # it is a different langauge, we use different embedding
        # as we don't have pretrained embedding layer for japanese words, we just apply built-in one:
        decoder_embedding = Embedding(WSeq2Seq.decoder_input_vocab_size, WSeq2Seq.EMBEDDING_DIM)
        decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
        decoder_lstm = LSTM(WSeq2Seq.LATENT_DIM, return_sequences=True, return_state=True, dropout=0.5)
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs_x,
            initial_state=encoder_states
        )

        decoder_dense = Dense(WSeq2Seq.decoder_input_vocab_size, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        # this model is just for the purpose of training the middle lstm layers (of encoder and decoder)
        # it will not be exported
        model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)

        model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["acc"]
        )

        r = model.fit(
            [WSeq2Seq.padded_encoder_inputs, WSeq2Seq.padded_decoder_inputs], WSeq2Seq.decoder_one_hot_target,
            batch_size=WSeq2Seq.BATCH_SIZE,
            epochs=WSeq2Seq.EPOCHS,
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

        encoder_model = Model(encoder_inputs_placeholder, encoder_states)

        decoder_state_input_h = Input(shape=(WSeq2Seq.LATENT_DIM,))
        decoder_state_input_c = Input(shape=(WSeq2Seq.LATENT_DIM,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

        decoder_outputs, h, c = decoder_lstm(
            decoder_inputs_single_x,
            initial_state=decoder_states_inputs
        )

        decoder_states = [h, c]
        decoder_outputs = decoder_dense(decoder_outputs)

        decoder_model = Model(
            [decoder_inputs_single] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )

        WSeq2Seq.encoder_model = encoder_model
        WSeq2Seq.decoder_model = decoder_model

    @staticmethod
    def decode_sequence(input_seq):
        states_value = WSeq2Seq.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = WSeq2Seq.decoder_input_word_to_index["<sos>"]
        eos_index = WSeq2Seq.decoder_input_word_to_index["<eos>"]

        output_sentence = []
        for _ in range(WSeq2Seq.max_decoder_input_seq_length):
            output_tokens, h, c = WSeq2Seq.decoder_model.predict(
                [target_seq] + states_value
            )

            idx = np.argmax(output_tokens[0, 0, :])

            if idx == eos_index:
                break

            word = ""
            if idx > 0:
                word = WSeq2Seq.index_to_word_trans[idx]
                output_sentence.append(word)

            target_seq[0, 0] = idx
            states_value = [h, c]

        return " ".join(output_sentence)

    @staticmethod
    def random_translation():
        while True:
            i = np.random.choice(len(WSeq2Seq.encoder_texts_input))
            input_seq = WSeq2Seq.padded_encoder_inputs[i:i+1]
            translation = WSeq2Seq.decode_sequence(input_seq)
            print("-")
            print("Input: ", WSeq2Seq.encoder_texts_input[i])
            print("Translation: ", translation)

            ans = input("Continue? [Y/n]")

            if ans and ans.lower().startswith("n"):
                break


WSeq2Seq.initialize()
WSeq2Seq.build_encoder_decoder_model()
WSeq2Seq.random_translation()
WSeq2Seq.initialize()
WSeq2Seq.build_encoder_decoder_model()
WSeq2Seq.random_translation()
