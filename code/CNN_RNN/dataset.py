import tensorflow as tf


def create_dataset(data, label, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices(
        (data, label)).batch(batch_size)

    return dataset
