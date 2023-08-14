import tensorflow as tf
import numpy as np

from argparse import ArgumentParser
from model import create_encoder, create_model
from utils import preprocessing


def predict(model, sentence):
    pred = model.predict(np.array([sentence]))
    return np.argmax(pred, axis=1)


def main():
    parser = ArgumentParser()
    parser.add_argument("sentence")
    parser.add_argument("model", default="model.h5")

    args = parser.parse_args()
    sentence = preprocessing(args.sentence)

    encoder = create_encoder()
    model = create_model(encoder=encoder)

    model.load_weights(parser.save)

    print(predict(model=args.model, sentence=sentence))


if __name__ == "__main__":
    main()
