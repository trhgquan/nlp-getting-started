# Natural Language Processing with Disaster Tweets
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

This project is licensed under [The GNU GPL v3](LICENSE)

## Notebooks 

*(using different models in different versions, please have a look at the version history)*

- [Statistical models notebook](https://www.kaggle.com/code/trhgquan/disaster-tweet-tfidf)
- [Deep learning notebooks](https://www.kaggle.com/code/trhgquan/disaster-tweet-with-llms)

## Code

View training & testing script's help with this command:
```
python <script>.py --help
```

**Note**: use those scripts at your own risk, since I don't normally re-train models on my personal PC.

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

| Parameter  | Value |
| ---------- | ----- |
| Train:test | 8:2   |
| Vocab size | 1000  |
| Epochs     | 10    |

#### LLMs

| Parameter      | Value    |
| -------------- | -------- |
| Train:dev:test | 6:2:2    |
| Batch size     | 64       |
| Learning rate  | 2e-5     |
| Weight decay   | 0.01     |
| Epochs         | 50       |
| Early stopping | 5 epochs |

#### Too-large LLMs

| Parameter      | Value    |
| -------------- | -------- |
| Train:dev:test | 6:2:2    |
| Batch size     | 32       |
| Learning rate  | 1e-5     |
| Weight decay   | 0.01     |
| Epochs         | 50       |
| Early stopping | 5 epochs |

### Results

*All experiments were conducted under the same random seed (42)*

[wandb.ai report](https://api.wandb.ai/links/khongsomeo/5rxjwfn6)

#### Statistical models

| Model                         | Training stats             | Public F1 |
| ----------------------------- | -------------------------- | --------- |
| SVC + TFIDF + CountVectorizer | [[1]](#statistical-models) | 0.80140   |
| SVC + TFIDF                   | [[1]](#statistical-models) | 0.80140   |
| RF + TFIDF + CountVectorizer  | [[1]](#statistical-models) | 0.78792   |


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
    <td>0.81520</td>
  </tr>
  <tr>
    <td rowspan="4">BART</td>
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
    <td><a href="https://huggingface.co/facebook/bart-large-mnli">large-mnli</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>(todo)</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/facebook/bart-large-cnn">large-cnn</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>(todo)</td>
  </tr>
  <tr>
    <td rowspan="4">BERT</td>
    <td><a href="https://huggingface.co/bert-base-uncased">base uncased</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.82899</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-large-uncased">base cased</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>(todo)</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-large-uncased">large uncased</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.83052</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-large-uncased">large cased</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>(todo)</td>
  </tr>
  <tr>
    <td>BERTweet</td>
    <td><a href="https://huggingface.co/vinai/bertweet-large">large</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.82899</td>
  </tr>
  <tr>
    <td>BORT</td>
	<td><a href="https://huggingface.co/amazon/bort">base</a></td>
	<td><a href="#LLMS">[2]</a></td>
	<td>0.74563</td>
  </tr>
  <tr>
    <td rowspan="2">DeBERTa v3</td>
    <td><a href="https://huggingface.co/microsoft/deberta-v3-base">base</a></td>
    <td><a href="#LLMS">[2]</a></td>
    <td>0.83205</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/microsoft/deberta-v3-large">large</a></td>
    <td><a href="#too-large-llms">[4]</a></td>
    <td>0.83113</td>
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
    <td>0.82163</td>
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
    <td>0.82500</td>
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