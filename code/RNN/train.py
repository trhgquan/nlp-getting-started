#!/bin/python3

from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split

from model import create_encoder, create_model
from utils import clean_df
from dataset import create_dataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--train_size", default=.8)
    parser.add_argument("--test_size", default=.2)
    parser.add_argument("--vocab_size", default=10000)
    parser.add_argument("--model_layers", default=2)
    parser.add_argument("--rnn_type", default="lstm")
    parser.add_argument("--rnn_units", nargs="*",
                        action="append", default=[64, 32])
    parser.add_argument("--dropout_rate", default=.5)
    parser.add_argument("--learning_rate", default=1e-4)
    parser.add_argument("--dense_units", default=64)
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--validation_steps", default=30)
    parser.add_argument("--save", default="model.h5")

    args = parser.parse_args()

    df = pd.read_csv(args.train)

    df = clean_df(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].tolist(),
        df['target'].tolist(),
        train_size=args.train_size,
        test_size=args.test_size,
        shuffle=True,
        random_state=42
    )

    encoder = create_encoder(X_train=X_train, VOCAB_SIZE=args.vocab_size)

    batch_size = args.batch_size
    rnn_type, rnn_units = args.rnn_type, args.rnn_units
    learning_rate, dropout_rate = args.learning_rate, args.dropout_rate
    model_layers, dense_units = args.model_layers, args.dense_units

    train_ds = create_dataset(
        data=X_train, label=y_train, batch_size=batch_size)
    val_ds = create_dataset(data=X_test, label=y_test,
                            batch_size=batch_size//2)

    model = create_model(encoder=encoder,
                         rnn_type=rnn_type,
                         rnn_units=rnn_units,
                         model_layers=model_layers,
                         dense_units=dense_units,
                         dropout_rate=dropout_rate,
                         learning_rate=learning_rate)

    hist = model.fit(train_ds, epochs=args.epochs, validation_data=(
        val_ds), validation_steps=args.validation_steps)

    print(f"Train loss: {hist['loss']}")
    print(f"Train accuracy: {hist['acc']}")

    model.save_weights(args.save)

    test_loss, test_acc = model.evaluate(val_ds)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    print("Finished training")


if __name__ == "__main__":
    main()
