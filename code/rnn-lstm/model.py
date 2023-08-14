import tensorflow as tf


def create_encoder(X_train=None, VOCAB_SIZE=1000):
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
    if X_train is not None:
        encoder.adapt(X_train)
    return encoder


def create_model(encoder):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optmizers.Adam(1e-4),
        metrics=["accuracy"]
    )

    return model
