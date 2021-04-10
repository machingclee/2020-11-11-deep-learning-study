import os
import sys
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

import numpy as np
import matplotlib.pyplot as plt


def softmax_over_time(x):
    assert(K.ndim(x) > 2)
    e = K.exp(x)
    s = K.sum(e, axis=1, keepdims=True)
    return e / s


class Attention:
    BATCH_SIZE = 64
    EPOCHS = 120
    LATENT_DIM = 256
    LATENT_DIM_DECODER = 256
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

    encoder_max_input_seq_length = 0
    decoder_max_input_seq_length = 0
    encoder_input_vocab_size = 0
    decoder_input_vocab_size = 0

    padded_encoder_inputs = []
    padded_decoder_inputs = []
    padded_decoder_outouts = []

    encoder_word_embedding_layer: Embedding = None
    decoder_one_hot_target = None

    encoder_model: Model = None
    decoder_model: Model = None

    attn_repeat_layer: RepeatVector = None
    attn_concat_layer: Concatenate = None
    attn_dense1: Dense = None
    attn_dense2: Dense = None
    attn_dot: Dot = None

    @staticmethod
    def initialize():
        Attention.generate_text_array()
        Attention.generate_encoder_input_tokenizer()
        Attention.generate_decoder_input_tokenizer()
        Attention.generate_encoder_input_word_to_index()
        Attention.generate_decoder_input_word_to_index()
        Attention.transform_texts_to_seqs()
        Attention.generate_input_vocab_sizes()
        Attention.generate_input_max_seq_lengths()
        Attention.generate_padded_seqs()
        Attention.generate_encoder_word_embedding_layer()
        Attention.generate_decoder_one_hot_target()
        Attention.generate_index_to_words()
        Attention.generate_attention_layers()

    @staticmethod
    def generate_text_array():
        t = 0
        for line in open("./jpn.txt", encoding="utf8"):
            t = t + 1
            if t > Attention.NUM_SAMPLES:
                break

            if "\t" not in line:
                continue

            values = line.split("\t")
            input_text = values[0]
            translation = values[1]

            target_text = translation + " <eos>"
            target_text_input = "<sos> " + translation

            Attention.encoder_texts_input.append(input_text)
            Attention.decoder_texts_output.append(target_text)
            Attention.decoder_texts_input.append(target_text_input)

    @staticmethod
    def generate_encoder_input_tokenizer():
        Attention.encoder_input_tokenizer = Tokenizer(num_words=Attention.MAX_VOCAB_SIZE, filters="")
        Attention.encoder_input_tokenizer.fit_on_texts(Attention.encoder_texts_input)

    @staticmethod
    def generate_decoder_input_tokenizer():
        Attention.decoder_input_tokenizer = Tokenizer(num_words=Attention.MAX_VOCAB_SIZE, filters="")
        # just to make sure both <sos> and <tos> are included, not efficient
        Attention.decoder_input_tokenizer.fit_on_texts(Attention.decoder_texts_output + Attention.decoder_texts_input)

    @staticmethod
    def generate_encoder_input_word_to_index():
        Attention.encoder_input_word_to_index = Attention.encoder_input_tokenizer.word_index

    @staticmethod
    def generate_decoder_input_word_to_index():
        Attention.decoder_input_word_to_index = Attention.decoder_input_tokenizer.word_index

    @staticmethod
    def transform_texts_to_seqs():
        Attention.encoder_input_seq = Attention.encoder_input_tokenizer.texts_to_sequences(Attention.encoder_texts_input)
        Attention.decoder_output_seq = Attention.decoder_input_tokenizer.texts_to_sequences(Attention.decoder_texts_output)
        Attention.decoder_input_seq = Attention.decoder_input_tokenizer.texts_to_sequences(Attention.decoder_texts_input)

    @staticmethod
    def generate_input_vocab_sizes():
        # +1 because we also include padding index 0 which correspond to nth in our vocab list
        # this becomes ncessary in creating embedding matrix
        # note that keras use word_count + 1 as the index of UNKNOWN, the key is "None" by default (already generated in word_index)
        Attention.encoder_input_vocab_size = len(Attention.encoder_input_tokenizer.word_index) + 1
        Attention.decoder_input_vocab_size = len(Attention.decoder_input_tokenizer.word_index) + 1

    @staticmethod
    def generate_input_max_seq_lengths():
        Attention.encoder_max_input_seq_length = max(len(s) for s in Attention.encoder_input_seq)
        Attention.decoder_max_input_seq_length = max(len(s) for s in Attention.decoder_input_seq)

    @staticmethod
    def generate_padded_seqs():
        Attention.padded_encoder_inputs = pad_sequences(
            Attention.encoder_input_seq,
            maxlen=Attention.encoder_max_input_seq_length
        )
        print("Attention.padded_encoder_inputs", Attention.padded_encoder_inputs.shape)
        # for LSTM training part input and output are of same length in sentences size
        Attention.padded_decoder_inputs = pad_sequences(
            Attention.decoder_input_seq,
            maxlen=Attention.decoder_max_input_seq_length,
            padding="post"
        )
        print("Attention.padded_decoder_inputs", Attention.padded_decoder_inputs.shape)
        Attention.padded_decoder_outouts = pad_sequences(
            Attention.decoder_output_seq,
            maxlen=Attention.decoder_max_input_seq_length,
            padding="post"
        )

    @staticmethod
    def generate_encoder_word_embedding_layer():
        print("Loading word vectors...")
        word_to_vec = {}
        with open(os.path.join("./glove.6B.{}d.txt".format(Attention.EMBEDDING_DIM)), encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.array(values[1:], dtype="float32")
                word_to_vec[word] = vec

        print("filling pre-trained embeddings")
        vocab_size = min(
            Attention.MAX_VOCAB_SIZE,
            len(Attention.encoder_input_word_to_index) + 1
        )
        embedding_matrix = np.zeros((vocab_size, Attention.EMBEDDING_DIM))

        # for embedding matrix, we are just interested in words in our training set:
        for word, index in Attention.encoder_input_word_to_index.items():
            word_vec = word_to_vec.get(word)
            if word_vec is not None:
                embedding_matrix[index] = word_vec

        Attention.encoder_word_embedding_layer = Embedding(
            vocab_size,
            Attention.EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=Attention.encoder_max_input_seq_length
        )

    @staticmethod
    def generate_decoder_one_hot_target():
        decoder_one_hot_targets = np.zeros(
            (
                len(Attention.decoder_texts_output),
                Attention.decoder_max_input_seq_length,
                Attention.decoder_input_vocab_size
            ),
            dtype="float32"
        )
        # e.g., decoder_targets_one_hot[0] is an array of binarized vectors, [e_1, e_4, ..., e_10], where e_k denotes the standard basis in R^num_words_output
        for i, wordDigits in enumerate(Attention.decoder_output_seq):
            for t, digit in enumerate(wordDigits):
                decoder_one_hot_targets[i, t, digit] = 1

        Attention.decoder_one_hot_target = decoder_one_hot_targets

    @staticmethod
    def generate_index_to_words():
        Attention.index_to_word_eng = {v: k for k, v in Attention.encoder_input_word_to_index.items()}
        Attention.index_to_word_trans = {v: k for k, v in Attention.decoder_input_word_to_index.items()}

    @staticmethod
    def generate_attention_layers():
        Attention.attn_repeat_layer = RepeatVector(Attention.encoder_max_input_seq_length)
        Attention.attn_concat_layer = Concatenate(axis=-1)
        Attention.attn_dense1 = Dense(10, activation="tanh")
        Attention.attn_dense2 = Dense(1, activation=softmax_over_time)
        Attention.attn_dot = Dot(axes=1)

    @staticmethod
    def one_step_attenion(h, s_prev):
        # h=[h_1,...,h_Tx], h.shape = (Tx, LATENT_DIM*2)
        # st_1 = s(t-1), st_1.shape = (LATENT_DIM_DECODER,)

        # copy s(t-1) Tx times:
        s_prev = Attention.attn_repeat_layer(s_prev)    # shape becomes (Tx, LATENT_DIM_DECODER)
        x = Attention.attn_concat_layer([h, s_prev])
        x = Attention.attn_dense1(x)
        alphas = Attention.attn_dense2(x)
        context = Attention.attn_dot([alphas, h])

        return context

    @staticmethod
    def stack_and_transpose(x):
        x = K.stack(x)
        x = K.permute_dimensions(x, pattern=(1, 0, 2))
        return x

    @staticmethod
    def build_encoder_decoder_model():
        encoder_inputs_placeholder = Input(
            shape=(Attention.encoder_max_input_seq_length,)
        )
        x = Attention.encoder_word_embedding_layer(encoder_inputs_placeholder)
        encoder = Bidirectional(LSTM(Attention.LATENT_DIM, return_sequences=True, dropout=0.5))
        encoder_outputs = encoder(x)

        decoder_inputs_placeholder = Input(shape=(Attention.decoder_max_input_seq_length,))
        decoder_embedding = Embedding(Attention.decoder_input_vocab_size, Attention.EMBEDDING_DIM)
        decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

        decoder_lstm = LSTM(Attention.LATENT_DIM_DECODER, return_state=True)
        decoder_dense = Dense(
            Attention.decoder_input_vocab_size,
            activation="softmax"
        )

        initial_s = Input(shape=(Attention.LATENT_DIM_DECODER,), name="s0")
        initial_c = Input(shape=(Attention.LATENT_DIM_DECODER,), name="c0")
        context_last_word_concat_layer = Concatenate(axis=2)

        s = initial_s
        c = initial_c

        outputs = []
        for t in range(Attention.decoder_max_input_seq_length):
            context = Attention.one_step_attenion(encoder_outputs, s)  # shape = (None, 2M_1)
            selector = Lambda(lambda x: x[:, t:t+1])
            x_curr = selector(decoder_inputs_x)
            decoder_lstm_input = context_last_word_concat_layer([context, x_curr])  # shape = (None, 2M_1 + EMBEDDING_DIM)
            o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

            decoder_outputs = decoder_dense(o)
            outputs.append(decoder_outputs)

        stacker = Lambda(Attention.stack_and_transpose)
        outputs = stacker(outputs)

        model = Model(
            inputs=[
                encoder_inputs_placeholder,
                decoder_inputs_placeholder,
                initial_s,
                initial_c
            ],
            outputs=outputs
        )

        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])

        z = np.zeros((Attention.NUM_SAMPLES, Attention.LATENT_DIM_DECODER))
        r = model.fit(
            [
                Attention.padded_encoder_inputs,
                Attention.padded_decoder_inputs,
                z,
                z
            ],
            Attention.decoder_one_hot_target,
            batch_size=Attention.BATCH_SIZE,
            epochs=Attention.EPOCHS,
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

        Attention.encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)

        encoder_outputs_as_input = Input(
            shape=(Attention.encoder_max_input_seq_length, Attention.LATENT_DIM*2,)
        )

        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

        context = Attention.one_step_attenion(encoder_outputs_as_input, initial_s)

        decoder_lstm_input = context_last_word_concat_layer(
            [context, decoder_inputs_single_x]
        )

        o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
        decoder_outputs = decoder_dense(o)

        Attention.decoder_model = Model(
            inputs=[
                decoder_inputs_single,
                encoder_outputs_as_input,
                initial_s,
                initial_c
            ],
            outputs=[decoder_outputs, s, c]
        )

    @staticmethod
    def decode_sequence(input_seq):
        encoder_out = Attention.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))

        target_seq[0, 0] = Attention.decoder_input_word_to_index["<sos>"]
        eos = Attention.decoder_input_word_to_index["<eos>"]

        s = np.zeros((1, Attention.LATENT_DIM_DECODER))
        c = np.zeros((1, Attention.LATENT_DIM_DECODER))

        output_seq = []
        for _ in range(Attention.decoder_max_input_seq_length):
            o, s, c = Attention.decoder_model.predict([target_seq, encoder_out, s, c])

            idx = np.argmax(o.flatten())

            if eos == idx:
                break

            word = ""
            if idx > 0:
                word = Attention.index_to_word_trans[idx]
                output_seq.append(word)

            target_seq[0, 0] = idx

        return " ".join(output_seq)

    @staticmethod
    def random_translation():
        while True:
            i = np.random.choice(len(Attention.encoder_texts_input))
            input_seq = Attention.padded_encoder_inputs[i:i+1]
            translation = Attention.decode_sequence(input_seq)
            print("-")
            print("Input: ", Attention.encoder_texts_input[i])
            print("Translation: ", translation)

            ans = input("Continue? [Y/n]")

            if ans and ans.lower().startswith("n"):
                break

    @staticmethod
    def custom_translation():
        while True:
            ans = input("Please input a sentence to translate:")
            input_seq = pad_sequences(
                Attention.encoder_input_tokenizer.texts_to_sequences([ans]),
                maxlen=Attention.encoder_max_input_seq_length
            )
            translation = Attention.decode_sequence(input_seq)
            print("-")
            print("translation:", translation)

            ans = input("Continue? [Y/n]")
            if ans and ans.lower().startswith("n"):
                break


Attention.initialize()
Attention.build_encoder_decoder_model()
Attention.custom_translation()
