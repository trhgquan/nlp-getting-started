# Natural Language Processing with Disaster Tweets
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

This project is licensed under [The GNU GPL v3](LICENSE)

## Notebooks 

*(using different models in different versions, please a check through the version history)*

- [Statistical models notebook](https://www.kaggle.com/code/trhgquan/disaster-tweet-tfidf)
- [Deep learning notebooks](https://www.kaggle.com/code/trhgquan/disaster-tweet-with-llms)

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

*All experiments were conducted under the same random seed (42)*

#### Statistical models

| Model                         | Training stats | Public scores |
| ----------------------------- | -------------- | ------------- |
| SVC + TFIDF + CountVectorizer | [^3]           | 0.80140       |
| SVC + TFIDF                   | [^3]           | 0.80140       |
| RF + TFIDF + CountVectorizer  | [^3]           | 0.78792       |


#### Deep learning models

| Model                                                                                                                                       | Training stats | Public F1   |
| ------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ----------- |
| [AlBERT v2 (base)](https://huggingface.co/albert-base-v2)                                                                                   | [^2]           | 0.79528     |
| [AlBERT v2 (large)](https://huggingface.co/albert-base-v2)                                                                                  | [^2]           | (todo)      |
| [BART (base)](https://huggingface.co/facebook/bart-base)                                                                                    | [^2]           | 0.82684     |
| [BART (large)](https://huggingface.co/facebook/bart-large)                                                                                  | [^2]           | 0.83726     |
| [BERT (base uncased)](https://huggingface.co/bert-base-uncased)                                                                             | [^2]           | 0.82899     |
| [BERT (large uncased)](https://huggingface.co/bert-large-uncased)                                                                           | [^2]           | 0.83052     |
| [BERTweet (large)](https://huggingface.co/vinai/bertweet-large)                                                                             | [^2]           | 0.82899     |
| [DeBERTa v3 (base)](https://huggingface.co/microsoft/deberta-v3-base)                                                                       | [^2]           | 0.83205     |
| [DistilBERT (base uncased)](https://huggingface.co/distilbert-base-uncased)                                                                 | [^2]           | 0.82439     |
| [RoBERTa (base)](https://huggingface.co/roberta-base)                                                                                       | [^2]           | 0.82868     |
| [RoBERTa (large)](https://huggingface.co/roberta-large)                                                                                     | [^2]           | **0.84033** |
| [Twitter RoBERTa Sentiment (base - latest)](https://huggingface.co/https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | [^2]           | 0.82776     |
| [Twitter RoBERTa (2021 - 124M)](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m)                                           | [^2]           | 0.83083     |
| [XLM-RoBERTa (base)](https://huggingface.co/xlm-roberta-base)                                                                               | [^2]           | 0.82439     |
| [XLNet (base cased)](https://huggingface.co/xlnet-base-cased)                                                                               | [^2]           | 0.82592     |
| BiLSTM Seq2Seq                                                                                                                              | [^1]           | 0.78302     |

There are several more LLMs that's worth trying (e.g. DeBERTa v3 large, ..etc) but Kaggle's RAM cannot handle those big boi. You can try training & finetuning them in a more powerful platform.

[^1]: Train size = 0.8, vocab size = 1000, training with 10 epochs.
[^2]: Train size = 0.6, learning rate 2e-5, weight decay 0.01, training with 50 epochs, early stopping after 5 epochs.
[^3]: Full training set

