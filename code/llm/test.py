#!/bin/python3

from argparse import ArgumentParser

import numpy as np
from model import create_model
from utils import clean_sentence


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

	if args.model == "vinai/bertweet-large":
		sentence = normalizeTweet(args.sentence)
	else:
	    sentence = clean_sentence(args.sentence)

    print("Predicting..")
    predict = np.argmax(model.predict(sentence), axis=1)
    print(predict)


if __name__ == "__main__":
    main()
