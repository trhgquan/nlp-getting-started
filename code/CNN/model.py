import tensorflow as tf
import numpy as np
import gensim.downloader as api


def create_encoder(X_train=None, VOCAB_SIZE=1000, sequence_length=30):
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=sequence_length,
        pad_to_max_tokens=True
    )

    if X_train is not None:
        encoder.adapt(X_train)
    return encoder


class CNN(tf.keras.Model):
    def __init__(self,
                 encoder=None,
                 filters=[3, 4, 5],
                 num_filters=100,
                 embedding_dim=300,
                 embedding_weights=None,
                 embedding_trainable=False,
                 sequence_length=30,
                 dropout_rate=.5):
        super().__init__()

        self.filters = filters
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        self.embedding_weights = embedding_weights
        self.embedding_trainable = embedding_trainable
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate

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
                kernel_regularizer=tf.keras.regularizers.L2(3)
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

        dropout = self.dropout(flatten)

        dense = self.dense(dropout)

        return self.final(dense)


def create_model(encoder,
                 num_filters=100,
                 filters=[3, 4, 5],
                 embedding_dim=300,
                 embedding_name=None,
                 embedding_trainable=False,
                 sequence_length=30,
                 dropout_rate=.5,
                 learning_rate=1e-4):

    if embedding_name is not None:
        embedding_dim, embedding_weights = load_embedding(encoder=encoder,
                                                          embedding_name=embedding_name)
    else:
        embedding_dim, embedding_weights = embedding_dim, None

    model = CNN(encoder=encoder,
                num_filters=num_filters,
                filters=filters,
                embedding_dim=embedding_dim,
                embedding_weights=embedding_weights,
                embedding_trainable=embedding_trainable,
                sequence_length=sequence_length,
                dropout_rate=dropout_rate)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optmizers.Adam(learning_rate),
        metrics=["accuracy"]
    )

    return model


def load_embedding(encoder=None, embedding_name=None):
    embedding_model = api.load(embedding_name)

    embedding_dim = embedding_model.vector_size

    embedding_matrix = np.zeros((len(encoder.get_vocabulary()), embedding_dim))

    for i, word in enumerate(encoder.get_vocabulary()):
        if word in embedding_model:
            embedding_matrix[i] = embedding_model[word]

    return embedding_dim, embedding_matrix
