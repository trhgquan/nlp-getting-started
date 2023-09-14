import tensorflow as tf


def create_encoder(X_train=None, VOCAB_SIZE=1000):
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
    if X_train is not None:
        encoder.adapt(X_train)
    return encoder


class BidirectionalRNNModel(tf.keras.Model):
    def __init__(self, encoder, rnn_type="lstm", rnn_units=64, dense_units=64):
        super().__init__()

        assert rnn_type in ["gru", "lstm"]

        self.rnn_type = rnn_type
        self.rnn_units = rnn_units
        self.dense_units = dense_units

        self.encoder = encoder

        self.embedding = tf.keras.layers.Embedding(
            input_dim=len(self.encoder.get_vocabulary()),
            output_dim=self.rnn_units,
            mask_zero=True
        )

        if self.rnn_type == "lstm":
            rnn_cell = tf.keras.layers.LSTM(self.rnn_units)
        elif self.rnn_type == "gru":
            rnn_cell = tf.keras.layers.GRU(self.rnn_units)

        self.bidirectional = tf.keras.layers.Bidirectional(rnn_cell)

        self.d1 = tf.keras.layers.Dense(self.dense_units)

        self.d2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.encoder(x)
        x = self.embedding(x)
        x = self.bidirectional(x)
        x = self.d1(x)

        return self.d2(x)


class StackedBidirectionalRNNModel(tf.keras.Model):
    def __init__(self, encoder, rnn_type="lstm", rnn_units=[64, 32], dense_units=64, dropout_rate=.5):
        super().__init__()

        assert rnn_type in ["lstm", "gru"]
        assert isinstance(rnn_units, list)

        self.rnn_type = rnn_type
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.encoder = encoder

        self.embedding = tf.keras.layers.Embedding(
            len(self.encoder.get_vocabulary()),
            self.rnn_units[0]
        )

        self.bidirectional_layers = []

        if self.rnn_type == "lstm":
            base_cell = tf.keras.layers.LSTM
        elif self.rnn_type == "gru":
            base_cell = tf.keras.layers.GRU

        for unit in self.rnn_units[:-1]:
            self.bidirectional_layers.append(
                tf.keras.layers.Bidirectional(
                    base_cell(unit, return_sequences=True))
            )

        self.bidirectional_layers.append(
            tf.keras.layers.Bidirectional(base_cell(self.rnn_units[-1]))
        )

        self.d1 = tf.keras.layers.Dense(self.dense_units, activation="relu")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.d2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.encoder(x)
        x = self.embedding(x)

        for layer in self.bidirectional_layers:
            x = layer(x)

        x = self.d1(x)
        x = self.dropout(x)

        return self.d2(x)


def create_model(encoder, model_layers=2, rnn_type="lstm", rnn_units=[64, 32], dense_units=64, dropout_rate=.5, learning_rate=1e-4):
    if model_layers == 1:
        model = BidirectionalRNNModel(
            encoder=encoder,
            rnn_type=rnn_type,
            rnn_units=rnn_units[0],
            dense_units=dense_units
        )
    else:
        model = StackedBidirectionalRNNModel(
            encoder=encoder,
            rnn_type=rnn_type,
            rnn_units=rnn_units,
            dense_units=dense_units,
            dropout_rate=dropout_rate
        )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optmizers.Adam(learning_rate),
        metrics=["accuracy"]
    )

    return model
