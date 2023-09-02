# Natural Language Processing with Disaster Tweets
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

This project is licensed under [The GNU GPL v3](LICENSE)

## Notebooks

*Please have a look at the version history of each notebook.*

- [Statistical models notebook](https://www.kaggle.com/code/trhgquan/disaster-tweet-tfidf)
- [Deep learning notebook](https://www.kaggle.com/code/trhgquan/disaster-tweet-with-llms)

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
- [Twitter RoBERTa Sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) requires masking username and url as specific tokens.

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

| Parameter            | Value    |
| -------------------- | -------- |
| Train:dev:test ratio | 6:2:2    |
| Batch size           | 64       |
| Learning rate        | 2e-5     |
| Weight decay         | 0.01     |
| Epochs               | 50       |
| Early stopping       | 5 epochs |

#### Too-large LLMs

| Parameter            | Value    |
| -------------------- | -------- |
| Train:dev:test ratio | 6:2:2    |
| Batch size           | 32       |
| Learning rate        | 1e-5     |
| Weight decay         | 0.01     |
| Epochs               | 50       |
| Early stopping       | 5 epochs |

### Results

All experiments were conducted under the same [Kaggle environment](https://www.kaggle.com/code/bconsolvo/hardware-available-on-kaggle):

| Configuration | Value                                |
| ------------- | ------------------------------------ |
| CPU           | Intel Xeon 2.20 GHz CPU, 4vCPU cores |
| Memory        | 32 GB                                |
| GPU           | Tesla T4 (x2)                        |
| Random seed   | 42                                   |


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
        <th>Notes</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td rowspan="2">AlBERT v2</td>
        <td><a href="https://huggingface.co/albert-base-v2">base</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.79528</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/albert-large-v2">large</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.81520</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4">BART</td>
        <td><a href="https://huggingface.co/facebook/bart-base">base</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82684</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/facebook/bart-large">large</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83726</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/facebook/bart-large-mnli">large-mnli</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83450</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/facebook/bart-large-cnn">large-cnn</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82347</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4">BERT</td>
        <td><a href="https://huggingface.co/bert-base-uncased">base uncased</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82899</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/bert-base-cased">base cased</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.81060</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/bert-large-uncased">large uncased</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83052</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/bert-large-cased">large cased</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82194</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4">BERTweet</td>
        <td><a href="https://huggingface.co/vinai/bertweet-base">base</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83726</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/vinai/bertweet-covid19-base-uncased">covid19-base-uncased</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.84002</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/vinai/bertweet-covid19-base-cased">covid19-base-cased</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82960</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/vinai/bertweet-large">large</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82899</td>
        <td></td>
    </tr>
    <tr>
        <td>BORT</td>
        <td><a href="https://huggingface.co/amazon/bort">base</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.74563</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">DeBERTa</td>
        <td><a href="https://huggingface.co/microsoft/deberta-base">base</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.81642</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/deberta-large">large</a></td>
		<td><a href="#too-large-llms">[4]</a></td>
        <td>0.84308</td>
        <td>best result so far</td>
    </tr>
    <tr>
    <td rowspan="4">DeBERTa v3</td>
        <td><a href="https://huggingface.co/microsoft/deberta-v3-xsmall">xsmall</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.80815</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/deberta-v3-small">small</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82408</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/deberta-v3-base">base</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83205</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/deberta-v3-large">large</a></td>
        <td><a href="#too-large-llms">[4]</a></td>
        <td>0.82745</td>
        <td></td>
    </tr>
    <tr>
        <td>mDeBERTa-V3</td>
        <td><a href="https://huggingface.co/microsoft/mdeberta-v3-base">base</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82929</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">DistilBERT</td>
        <td><a href="https://huggingface.co/distilbert-base-uncased">base uncased</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82439</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/distilbert-base-cased">base cased</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82163</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">RoBERTa</td>
        <td><a href="https://huggingface.co/roberta-base">base</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82868</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/roberta-large">large</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.84033</td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="3">Twitter RoBERTa Sentiment</td>
        <td><a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment">base</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83389</td>
        <td rowspan="3"><a href="https://huggingface.co/cardiffnlp">CardiffNLP</a> has a huge list of Twitter pretrained models and these are just 3 of them. Try finetuning others (if you have time).</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest">base latest</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82776</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m">base 2021 124M</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83083</td>
    </tr>
    <tr>
        <td rowspan="2">XLM-RoBERTa</td>
        <td><a href="https://huggingface.co/xlm-roberta-base">base</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82439</td>
        <td></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/xlm-roberta-large">large</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82500</td>
        <td></td>
    </tr>
    <tr>
        <td>XLNet</td>
        <td><a href="https://huggingface.co/xlnet-base-cased">base cased</a></td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82592</td>
        <td></td>
    </tr>
    <tr>
        <td>BiLSTM Seq2Seq</td>
        <td></td>
        <td><a href="#deep-learning-models">[3]</a></td>
        <td>0.78302</td>
        <td></td>
    </tr>
</tbody>
</table>