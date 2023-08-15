#!/bin/python3

import numpy as np

from model import create_model
from utils import remove_emoji, remove_html, remove_punct, remove_URL
from argparse import ArgumentParser


def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = remove_emoji(sentence)
    sentence = remove_html(sentence)
    sentence = remove_punct(sentence)
    sentence = remove_URL(sentence)
    sentence = sentence.replace("\s+", " ", regex=True)

    return sentence


def main():
    parser = ArgumentParser()
    parser.add_argument("--sentence")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--num_labels", default=2)

    args = parser.parse_args()

    model = create_model(
        pretrain=args.model,
        num_labels=args.num_labels
    )

    sentence = clean_sentence(args.sentence)

    print("Predicting..")
    predict = np.argmax(model.predict(sentence), axis=1)
    print(predict)


if __name__ == "__main__":
    main()
