#!/bin/python3

from utils import clean_sentence
from argparse import ArgumentParser

import pickle as pkl


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--sentence")

    args = parser.parse_args()
    with open(args.model, "rb+") as f:
        pipeline = pkl.load(f)

    sentence = clean_sentence(args.sentence)
    print(f"Predict: {pipeline.predict(sentence)}")


if __name__ == "__main__":
    main()
