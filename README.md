# Natural Language Processing with Disaster Tweets
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

This project is licensed under [The GNU GPL v3](LICENSE)

## Baselines

### Text preprocessing
Different text preprocessing methods used in my implementations, but most methods following these steps

- Removing emojis
- Removing html
- Removing URLs
- Removing punctuations
- Lowercase and remove multiple spaces.

However there are some exceptions where a specific preprocessing method of the pretrained model is applied:

- [BERTweet](https://huggingface.co/vinai/bertweet-large) [using TweetTokenizer to mask and replace some tokens](https://github.com/VinAIResearch/BERTweet#-normalize-raw-input-tweets)
- [Twitter RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) requires masking username and url as specific tokens.

### Results

#### Statistical models

| Model                         | Training stats | Public scores |
| ----------------------------- | -------------- | ------------- |
| SVC + TFIDF                   | [^3]           | 0.80140       |
| SVC + TFIDF + CountVectorizer | [^3]           | 0.80140       |
| RF + TFIDF + CountVectorizer  | [^3]           | 0.78792       |


#### Deep learning models

| Model                                                                                          | Training stats | Public F1   |
| ---------------------------------------------------------------------------------------------- | -------------- | ----------- |
| Finetuning [Twitter RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) | [^2]           | **0.83083** |
| Finetuning [BERTweet](https://huggingface.co/vinai/bertweet-large)                             | [^2]           | 0.82899     |
| Finetuning [BERT](https://huggingface.co/bert-base-uncased)                                    | [^2]           | 0.82899     |
| Finetuning [RoBERTa](https://huggingface.co/roberta-base)                                      | [^2]           | 0.82868     |
| Finetuning [XLNet](https://huggingface.co/xlnet-base-cased)                                    | [^2]           | 0.82592     |
| Finetuning [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base)                              | [^2]           | 0.82439     |
| Finetuning [DistilBERT](https://huggingface.co/distilbert-base-uncased)                        | [^2]           | 0.82439     |
| Finetuning [AlBERTa](https://huggingface.co/albert-base-v2)                                    | [^2]           | 0.79528     |
| BiLSTM Seq2Seq                                                                                 | [^1]           | 0.78302     |

[^1]: Train size = 0.8, vocab size = 1000, training with 10 epochs.
[^2]: Train size = 0.6, learning rate 2e-5, weight decay 0.01, training with 50 epochs, early stopping after 5 epochs.
[^3]: Full training set

