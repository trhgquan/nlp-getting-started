# Natural Language Processing with Disaster Tweets
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

This project is licensed under [The GNU GPL v3](LICENSE)

## Baselines

| Model                  | Training stats | Public scores |
| ---------------------- | -------------- | ------------- |
| BiLSTM Seq2Seq         | [^1]           | 0.78302       |
| Finetuning BERT		 | [^2]           | 0.82899		  |
| Finetuning DistilBERT  | [^2]           | 0.82439       |
| Finetuning XLM-RoBERTa | [^2]           | 0.82439       |
| Finetuning AlBERTa     | [^2]			  | 0.79528       |

[^1]: Train size = 0.8, vocab size = 1000, training with 10 epochs.
[^2]: Train size = 0.6, learning rate 2e-5, weight decay 0.01, training with 50 epochs, early stopping after 5 epochs.


