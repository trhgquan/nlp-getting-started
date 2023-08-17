#!/bin/python3

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from argparse import ArgumentParser
from utils import clean_df
from model import create_pipeline
import pandas as pd
import pickle as pkl


def main():
    parser = ArgumentParser()
    parser.add_argument("--train", default="train.csv")
	parser.add_argument("--classifier", default="svc")
    parser.add_argument("--model_path", default="model.pkl")
    parser.add_argument("--tokenizer_path", default="tokenizer.pkl")

    args = parser.parse_args()
    df = pd.read_csv(args.train)
    train_df = clean_df(df)

    train_df_X, train_df_y = train_df["text"], train_df["target"]

	classifier = "svc"
	if args.classifier == "rf":
		classifier = RandomForestClassifier
	if args.classifier == "svc":
		classifier = SVC

    pipeline = create_pipeline(classifier = classifier)

    pipeline.fit(train_df_X, train_df_y)

    print(f"Score: {pipeline.score(train_df_X, train_df_y)}")

    with open(args.model_path, "wb+") as f:
        pkl.dump(f, pipeline)


if __name__ == "__main__":
    main()
