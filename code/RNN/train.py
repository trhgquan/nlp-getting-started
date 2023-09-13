#!/bin/python3

from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split

from model import create_encoder, create_model
from utils import preprocessing


def main():
    parser = ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--train_size", default=.8)
    parser.add_argument("--test_size", default=.2)
    parser.add_argument("--vocab_size", default=1000)
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--validation_steps", default=30)
    parser.add_argument("--save", default="model.h5")

    args = parser.parse_args()

    df = pd.read_csv(args.train)

    df["text"].apply(lambda x: preprocessing(x))

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].tolist(),
        df['target'].tolist(),
        train_size=args.train_size,
        test_size=args.test_size,
        shuffle=True,
        random_state=42
    )

    encoder = create_encoder(X_train=X_train, VOCAB_SIZE=args.vocab_size)

    model = create_model(encoder=encoder)

    hist = model.fit(X_train, y_train, epochs=args.epochs, validation_data=(
        X_test, y_test), validation_steps=args.validation_steps)

    print(f"Train loss: {hist['loss']}")
    print(f"Train accuracy: {hist['acc']}")

    model.save_weights(args.save)

    test_loss, test_acc = model.evaluate(X_test, y_test)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    print("Finished training")


if __name__ == "__main__":
    main()
