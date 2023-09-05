#!/bin/python3

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from dataset import DisasterDataset
from model import create_model, create_trainer
from sklearn.base import accuracy_score
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

from utils import clean_df, normalizeTweet

random_state = 42


def main():
    parser = ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--pretrain", default="distilbert-base-uncased")
    parser.add_argument("--save", default="model.pt")
    parser.add_argument("--num_labels", default=2)
    parser.add_argument("--train_size", default=.6)
    parser.add_argument("--dev_size", default=.5)
    parser.add_argument("--test_size", default=.5)
    parser.add_argument("--early_stopping_patience", default=5)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--evaluation_strategy", default="epoch")
    parser.add_argument("--save_strategy", default="epoch")
    parser.add_argument("--save_total_limit", default=2)
    parser.add_argument("--num_train_epochs", default=50)
    parser.add_argument("--per_device_train_batch_size", default=64)
    parser.add_argument("--per_device_eval_batch_size", default=64)
    parser.add_argument("--learning_rate", default=2e-5)
    parser.add_argument("--warmup_steps", default=500)
    parser.add_argument("--weight_decay", default=.01)
    parser.add_argument("--metric_for_best_model", default="accuracy")
    parser.add_argument("--load_best_model_at_end", default=True)
    parser.add_argument("--preprocessing_url_mode", default="remove_url")

    args = parser.parse_args()

    # Create model and tokenizer from pretrain

    tokenizer, model = create_model(
        pretrain=args.pretrain,
        num_labels=args.num_labels
    )

    # Read and create dataset
    train_df = pd.read_csv(args.train)

    # Normalize text
    if args.model == "vinai/bertweet-large":
        train_df["text"] = train_df["text"].apply(
            lambda x: normalizeTweet(x))
    else:
        train_df = clean_df(train_df, mode=args.preprocessing_url_mode)

    df_train, df_remain = train_test_split(
        train_df, test_size=1 - args.train_size, random_state=random_state, stratify=train_df["target"])
    df_dev, df_test = train_test_split(
        df_remain, test_size=args.test_size, random_state=random_state, stratify=df_remain["target"])

    train_dataset = DisasterDataset(
        encodings=tokenizer(df_train["text"].tolist(),
                            truncation=True, padding=True),
        labels=df_train["target"].tolist()
    )
    dev_dataset = DisasterDataset(
        encodings=tokenizer(df_dev["text"].tolist(),
                            truncation=True, padding=True),
        labels=df_dev["target"].tolist()
    )
    test_dataset = DisasterDataset(
        encodings=tokenizer(df_test["text"].tolist(),
                            truncation=True, padding=True),
        labels=df_test["target"].tolist()
    )

    # Droppiing last batch to fit the training
    if args.model == "xlnet-base-cased" or args.model == "xlnet-large-cased":
        dataloader_drop_last = True
    else:
        dataloader_drop_last = False

    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=args.load_best_model_at_end,
        dataloader_drop_last=dataloader_drop_last
    )

    print("Training")
    trainer.train()

    print("Evaluating")
    trainer.evaluate()

    print("Testing")
    predictions, labels, _ = trainer.predict(test_dataset)

    predictions = np.argmax(predictions, axis=1)

    print(classification_report(predictions, labels))

    print(f"Accuracy = {accuracy_score(predictions, labels):.6f}, \
        Precision = {precision_score(predictions, labels, average = 'macro'):.6f}, \
        Recall = {recall_score(predictions, labels, average = 'macro'):.6f}, \
        F1 = {f1_score(predictions, labels, average = 'macro'):.6f}")

    print("Saving pretrained")
    trainer.save_model(args.save)


if __name__ == "__main__":
    main()
