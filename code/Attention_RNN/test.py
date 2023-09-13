#!/bin/python3

from argparse import ArgumentParser

import numpy as np
from model import create_encoder, create_model
from utils import clean_sentence


def predict(model, sentence):
    pred = model.predict(np.array([sentence]))
    return np.argmax(pred, axis=1)


def main():
    parser = ArgumentParser()
    parser.add_argument("sentence")
    parser.add_argument("model", default="model.h5")

    args = parser.parse_args()
    sentence = clean_sentence(args.sentence)

    encoder = create_encoder()
    model = create_model(encoder=encoder)

    model.load_weights(parser.save)

    print(predict(model=args.model, sentence=sentence))


if __name__ == "__main__":
    main()
