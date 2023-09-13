import tensorflow as tf
import numpy as np


class Attention(tf.keras.Model):
    def __init__(self, units, attention_type="dot"):
        super(Attention, self).__init__()

        self.attention_type = attention_type
        self.units = units

        if self.attention_type == "general":
            self.W1 = tf.keras.layers.Dense(self.units * 2)

        elif self.attention_type == "concat":
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        if self.attention_type == "dot":
            hidden_with_time_axis = tf.expand_dims(hidden, axis=-1)
            score = tf.matmul(features, hidden_with_time_axis)
            attention_weights = tf.nn.softmax(score, axis=1)

        elif self.attention_type == "general":
            hidden_with_time_axis = tf.expand_dims(hidden, axis=-1)
            score = tf.matmul(self.W1(features), hidden_with_time_axis)
            attention_weights = tf.nn.softmax(score, axis=1)

        elif self.attention_type == "concat":
            hidden_with_time_axis = tf.expand_dims(hidden, axis=1)
            score = tf.nn.tanh(self.W1(features) +
                               self.W2(hidden_with_time_axis))
            attention_weights = tf.nn.softmax(self.V1(score), axis=1)

        context_vector = attention_weights * features

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class RNN_with_Attention(tf.keras.Model):
    def __init__(self, encoder, dense_units=64, attention_type="dot", attention_units=64, cells_type="gru", cells_unit=64):
        super(RNN_with_Attention, self).__init__()

        assert attention_type in ["dot", "general", "concat"]
        assert cells_type in ["gru", "lstm"]

        self.encoder = encoder
        self.attention_type = attention_type
        self.attention_unit = attention_units
        self.cells_type = cells_type
        self.cells_unit = cells_unit

        print(
            f"Initializing model with {self.attention_type} attention and {self.cells_type} cells")

        self.embedding = tf.keras.layers.Embedding(
            input_dim=len(self.encoder.get_vocabulary()),
            output_dim=self.cells_unit,
            mask_zero=True,
            name="embedding"
        )

        if self.cells_type == "gru":
            bidirectional_cell = tf.keras.layers.GRU(
                self.cells_unit,
                return_sequences=True,
                return_state=True
            )

        elif self.cells_type == "lstm":
            bidirectional_cell = tf.keras.layers.LSTM(
                self.cells_unit,
                return_sequences=True,
                return_state=True
            )

        self.bidirectional = tf.keras.layers.Bidirectional(
            bidirectional_cell,
            name="bidirectional_rnn"
        )

        self.attention = Attention(
            units=self.attention_units, attention_type=self.attention_type)
        self.d1 = tf.keras.layers.Dense(self.dense_units, activation="sigmoid")
        self.d2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        enc = self.encoder(inputs)
        emb = self.embedding(enc)

        if self.cell_type == "gru":
            output, forward_final_state, backward_final_state = self.bidirectional(
                emb)
            final_state = tf.keras.layers.Concatenate()(
                [forward_final_state, backward_final_state])

        elif self.cell_type == "lstm":
            output, forward_h, forward_c, backward_h, backward_c = self.bidirectional(
                emb)
            final_state = tf.keras.layers.Concatenate()(
                [forward_h, backward_h])

        context_vector, attention_weights = self.attention(output, final_state)

        d1 = self.d1(context_vector)

        return self.d2(d1)


def create_encoder(X_train, VOCAB_SIZE=10000):
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(X_train)

    return encoder


def create_model(encoder, dense_units=64, rnn_type="gru", rnn_units=64, attention_type="dot", attention_units=64, learning_rate=1e-4):
    model = RNN_with_Attention(encoder=encoder,
                               dense_units=dense_units,
                               attention_units=attention_units,
                               attention_type=attention_type,
                               rnn_units=rnn_units,
                               rnn_type=rnn_type)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=["accuracy"])

    print(model.summary())

    return model
