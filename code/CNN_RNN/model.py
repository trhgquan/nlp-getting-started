import tensorflow as tf
import gensim.downloader as api
import numpy as np


def create_encoder(X_train=None, VOCAB_SIZE=1000, sequence_length=30):
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=sequence_length,
        pad_to_max_tokens=True
    )

    if X_train is not None:
        encoder.adapt(X_train)
    return encoder


def load_embedding(encoder=None, embedding_name=None):
    embedding_model = api.load(embedding_name)

    embedding_dim = embedding_model.vector_size

    embedding_matrix = np.zeros((len(encoder.get_vocabulary()), embedding_dim))

    for i, word in enumerate(encoder.get_vocabulary()):
        if word in embedding_model:
            embedding_matrix[i] = embedding_model[word]

    return embedding_dim, embedding_matrix


class CNN_RNN(tf.keras.Model):
    def __init__(self,
                 encoder=None,
                 filters=[3, 4, 5],
                 num_filters=100,
                 embedding_dim=300,
                 embedding_weights=None,
                 embedding_trainable=False,
                 recurrent_units=512,
                 recurrent_type="lstm",
                 sequence_length=30,
                 dropout_rate=.5):
        super().__init__()

        recurrent_type = recurrent_type.lower()

        assert recurrent_type in ["lstm", "bilstm", "gru", "bigru"]

        self.filters = filters
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.recurrent_units = recurrent_units
        self.recurrent_type = recurrent_type
        self.embedding_weights = embedding_weights
        self.embedding_trainable = embedding_trainable

        self.encoder = encoder

        if self.embedding_weights is not None:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                trainable=self.embedding_trainable,
                weights=[self.embedding_weights],
                name="embedding"
            )
        else:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                trainable=self.embedding_trainable,
                name="embedding"
            )

        self.reshape = tf.keras.layers.Reshape(
            (self.sequence_length, self.embedding_dim, 1)
        )

        self.convo = [
            tf.keras.layers.Convo2D(
                filters=self.num_filters,
                kernel_size=(i, self.embedding_dim),
                activation="relu",
            ) for i in self.filters
        ]

        self.max_pools = [
            tf.keras.layers.MaxPooling2D(
                pool_size=(self.sequence_length - i + 1, 1),
                strides=(1, 1),
                padding="valid"
            ) for i in self.filters
        ]

        self.concat = tf.keras.layers.Concatenate(axis=1)

        self.flatten = tf.keras.layers.Flatten()

        if self.recurrent_type == "lstm":
            self.recurrent = tf.keras.layers.LSTM(
                units=self.recurrent_units,
                return_state=True
            )

        elif self.recurrent_type == "bilstm":
            self.recurrent = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=self.recurrent_units,
                    return_state=True
                )
            )

        elif self.recurrent_type == "gru":
            self.recurrent = tf.keras.layers.GRU(
                units=self.recurrent_units,
                return_state=True
            )

        elif self.recurrent_type == "bigru":
            self.recurrent = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=self.recurrent_units,
                    return_state=True
                )
            )

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.dense = tf.keras.layers.Dense(64, activation="sigmoid")

        self.final = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.encoder(x)
        x = self.embedding(x)
        x = self.reshape(x)

        convo_output = [convo(x) for convo in self.convo]
        maxpool_output = [self.max_pools[i](convo_output[i])
                          for i in range(len(self.filters))]

        concat = self.concat(maxpool_output)

        flatten = self.flatten(concat)

        flatten_expanded = tf.expand_dims(flatten, axis=-1)

        if self.recurrent_type == "lstm":
            recurrent_output, state_h, _ = self.recurrent(flatten_expanded)
        elif self.recurrent_type == "gru":
            recurrent_output, state = self.recurrent(flatten_expanded)

        dropout = self.dropout(recurrent_output)

        dense = self.dense(dropout)

        final = self.final(dense)

        return final


class CNN_RNN_Concat(CNN_RNN):
    def __init__(self,
                 encoder=None,
                 filters=[3, 4, 5],
                 num_filters=100,
                 embedding_dim=300,
                 embedding_weights=None,
                 embedding_trainable=False,
                 recurrent_units=512,
                 recurrent_type="lstm",
                 sequence_length=30,
                 dropout_rate=.5):
        super().__init__()

        recurrent_type = recurrent_type.lower()

        assert recurrent_type in ["lstm", "bilstm", "gru", "bigru"]

        self.filters = filters
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.recurrent_units = recurrent_units
        self.recurrent_type = recurrent_type
        self.embedding_weights = embedding_weights
        self.embedding_trainable = embedding_trainable

        self.encoder = encoder

        if self.embedding_weights is not None:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                trainable=self.embedding_trainable,
                weights=[self.embedding_weights],
                name="embedding"
            )
        else:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                trainable=self.embedding_trainable,
                name="embedding"
            )

        self.reshape = tf.keras.layers.Reshape(
            (self.sequence_length, self.embedding_dim, 1)
        )

        self.convo = [
            tf.keras.layers.Convo2D(
                filters=self.num_filters,
                kernel_size=(i, self.embedding_dim),
                activation="relu",
            ) for i in self.filters
        ]

        self.max_pools = [
            tf.keras.layers.MaxPooling2D(
                pool_size=(self.sequence_length - i + 1, 1),
                strides=(1, 1),
                padding="valid"
            ) for i in self.filters
        ]

        self.concat_cnn = tf.keras.layers.Concatenate(axis=1)

        self.flatten = tf.keras.layers.Flatten()

        if self.recurrent_type == "lstm":
            self.recurrent = tf.keras.layers.LSTM(
                units=self.recurrent_units,
                return_state=True
            )

        elif self.recurrent_type == "bilstm":
            self.recurrent = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=self.recurrent_units,
                    return_state=True
                )
            )

        elif self.recurrent_type == "gru":
            self.recurrent = tf.keras.layers.GRU(
                units=self.recurrent_units,
                return_state=True
            )

        elif self.recurrent_type == "bigru":
            self.recurrent = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=self.recurrent_units,
                    return_state=True
                )
            )

        self.concat_rnn_cnn = tf.keras.layers.Concatenate(axis=1)

        self.dense = tf.keras.layers.Dense(64, activation="sigmoid")

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.final = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.encoder(x)
        x = self.embedding(x)
        x_cnn = self.reshape(x)

        convo_output = [convo(x_cnn) for convo in self.convo]

        maxpool_output = [self.max_pools[i](convo_output[i])
                          for i in range(len(self.filters))]

        concat_cnn = self.concat_cnn(maxpool_output)

        flatten_cnn = self.flatten(concat_cnn)

        if self.recurrent_type == "lstm":
            recurrent_output, state_h, _ = self.recurrent(x)
        elif self.recurrent_type == "gru":
            recurrent_output, state = self.recurrent(x)

        concat_cnn_rnn = self.concat_cnn_rnn([flatten_cnn, recurrent_output])

        dense = self.dense(concat_cnn_rnn)

        dropout = self.dropout(dense)

        final = self.final(dropout)

        return final


def create_model(model_type="feed",
                 encoder=None,
                 num_filters=100,
                 filters=[3, 4, 5],
                 embedding_dim=300,
                 embedding_name=None,
                 embedding_trainable=False,
                 sequence_length=30,
                 recurrent_units=512,
                 recurrent_type="lstm",
                 dropout_rate=.5,
                 learning_rate=1e-4):

    assert model_type in ["feed", "concat"]

    if embedding_name is not None:
        embedding_weights, embedding_dim = load_embedding(encoder=encoder,
                                                          embedding_name=embedding_name)

    if model_type == "feed":
        model = CNN_RNN(encoder=encoder,
                        num_filters=num_filters,
                        filters=filters,
                        embedding_dim=embedding_dim,
                        embedding_weights=embedding_weights,
                        embedding_trainable=embedding_trainable,
                        recurrent_units=recurrent_units,
                        recurrent_type=recurrent_type,
                        sequence_length=sequence_length,
                        dropout_rate=dropout_rate)

    elif model_type == "concat":
        model = CNN_RNN_Concat(encoder=encoder,
                               num_filters=num_filters,
                               filters=filters,
                               embedding_dim=embedding_dim,
                               embedding_weights=embedding_weights,
                               embedding_trainable=embedding_trainable,
                               recurrent_units=recurrent_units,
                               recurrent_type=recurrent_type,
                               sequence_length=sequence_length,
                               dropout_rate=dropout_rate)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optmizers.Adam(learning_rate),
        metrics=["accuracy"]
    )

    return model
