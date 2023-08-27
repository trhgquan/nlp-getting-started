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

### Training configurations

#### Statistical models
Using full training set.
  
#### Deep learning models
Train size = 0.8, vocab size = 1000, training with 10 epochs.

#### LLMs
Train size = 0.6, learning rate 2e-5, weight decay 0.01, training with 50 epochs, early stopping after 5 epochs.

### Results

*All experiments were conducted under the same random seed (42)*

#### Statistical models

| Model                         | Training stats             | Public scores |
| ----------------------------- | -------------------------- | ------------- |
| SVC + TFIDF + CountVectorizer | [[1]](#statistical-models) | 0.80140       |
| SVC + TFIDF                   | [[1]](#statistical-models) | 0.80140       |
| RF + TFIDF + CountVectorizer  | [[1]](#statistical-models) | 0.78792       |


#### Deep learning models

<table>
<thead>
  <tr>
    <th colspan="2">Model</th>
    <th>Training configurations</th>
    <th>Public F1</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">AlBERT v2</td>
    <td><a href="https://huggingface.co/albert-base-v2">base</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.79528</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/albert-large-v2">large</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>(todo)</td>
  </tr>
  <tr>
    <td rowspan="2">BART</td>
    <td><a href="https://huggingface.co/facebook/bart-base">base</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.82684</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/facebook/bart-large">large</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.83726</td>
  </tr>
  <tr>
    <td rowspan="2">BERT</td>
    <td><a href="https://huggingface.co/bert-base-uncased">base uncased</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.82899</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-large-uncased">large uncased</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.83052</td>
  </tr>
  <tr>
    <td>BERTweet</td>
    <td><a href="https://huggingface.co/vinai/bertweet-large">large</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.82899</td>
  </tr>
  <tr>
    <td>DeBERTa v3</td>
    <td><a href="https://huggingface.co/microsoft/deberta-v3-base">base</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.83205</td>
  </tr>
  <tr>
    <td rowspan="2">DistilBERT</td>
    <td><a href="https://huggingface.co/distilbert-base-uncased">base uncased</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.82439</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/distilbert-base-cased">base cased</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>(todo)</td>
  </tr>
  <tr>
    <td rowspan="2">RoBERTa</td>
    <td><a href="https://huggingface.co/roberta-base">base</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.82868</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/roberta-large">large</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.84033</td>
  </tr>
  <tr>
    <td>Twitter RoBERTa Sentiment</td>
    <td><a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest">base latest</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.82776</td>
  </tr>
  <tr>
    <td>Twitter RoBERTa</td>
    <td><a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m">2021 - 144M</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.83083</td>
  </tr>
  <tr>
    <td rowspan="2">XLM-RoBERTa</td>
    <td><a href="https://huggingface.co/xlm-roberta-base">base</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.82439</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/xlm-roberta-large">large</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>(todo)</td>
  </tr>
  <tr>
    <td>XLNet</td>
    <td><a href="https://huggingface.co/xlnet-base-cased">base cased</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.82592</td>
  </tr>
  <tr>
    <td>BiLSTM Seq2Seq</td>
    <td></td>
    <td><a href="#deep-learning-models">[3]</a></td>
    <td>0.78302</td>
  </tr>
</tbody>
</table>

There are several more LLMs that's worth trying (e.g. DeBERTa v3 large, ..etc) but Kaggle's RAM cannot handle those big boi. You can try training & finetuning them in a more powerful platform.