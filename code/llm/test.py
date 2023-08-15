#!/bin/python3

import numpy as np

from model import create_model
from utils import clean_sentence
from argparse import ArgumentParser
              
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
