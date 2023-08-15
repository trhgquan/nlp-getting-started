import evaluate
import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)


def create_model(pretrain, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(pretrain)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrain, num_labels=num_labels)

    return tokenizer, model


def create_trainer(**kwargs):
    training_args = TrainingArguments(
        output_dir=kwargs.get("output_dir", "./results"),
        evaluation_strategy=kwargs.get("evaluation_strategy", "epoch"),
        save_strategy=kwargs.get("save_strategy", "epoch"),
        save_total_limit=kwargs.get("save_total_limit", 2),
        num_train_epochs=kwargs.get("num_train_epochs", 50),
        per_device_train_batch_size=kwargs.get(
            "per_device_train_batch_size", 64),
        per_device_eval_batch_size=kwargs.get(
            "per_device_eval_batch_size", 64),
        learning_rate=kwargs.get("learning_rate", 2e-5),
        warmup_steps=kwargs.get("warmup_steps", 500),
        weight_decay=kwargs.get("weight_decay", 0.01),
        metric_for_best_model=kwargs.get("metric_for_best_model", "accuracy"),
        load_best_model_at_end=kwargs.get("load_best_model_at_end", True)
    )

    glue_metric = evaluate.load("glue", "mnli")

    def compute_metrics(eval_pred):
        predictions, labels, _ = eval_pred

        predictions = np.argmax(predictions, axis=1)

        return glue_metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=kwargs.get("model", None),
        args=training_args,
        train_dataset=kwargs.get("train_dataset", None),
        eval_dataset=kwargs.get("eval_dataset", None),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=kwargs.get("early_stopping_patience", 5))]
    )

    return trainer
