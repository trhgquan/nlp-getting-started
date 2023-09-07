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

| Hyperparameter  | Value |
| --------------- | ----- |
| Train:test      | 8:2   |
| Vocab size      | 1000  |
| Epochs          | 10    |

#### LLMs

| Hyperparameter       | Value    |
| -------------------- | -------- |
| Train:dev:test ratio | 6:2:2    |
| Batch size           | 64       |
| Learning rate        | 2e-5     |
| Weight decay         | 0.01     |
| Epochs               | 50       |
| Early stopping       | 5 epochs |

#### Too-large LLMs

Some large LLMs cannot be trained with [hyperparameters in the LLMs section](#LLMs). In order to fit those models to Kaggle GPU's RAM, I reduced the batch size and learning rate to following values:

| Hyperparameter       | Value    |
| -------------------- | -------- |
| Train:dev:test ratio | 6:2:2    |
| Batch size           | 32       |
| Learning rate        | 1e-5     |
| Weight decay         | 0.01     |
| Epochs               | 50       |
| Early stopping       | 5 epochs |

All remaining hyperparametes stay the same as [LLMs](#LLMs).

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

<table>
<thead>
	<tr>
		<th colspan="2">Model (Vectorizer)</th>
		<th>Training configurations</th>
		<th>Public F1</th>
	</tr>
</thead>
<tbody>
	<tr>
		<td>SVC</td>
		<td></td>
		<td>TFIDF</td>
		<td><a href="#statistical-models">[1]</a></td>
		<td>0.80140</td>
	</tr>
	<tr>
		<td>Random Forest</td>
		<td></td>
		<td>TFIDF</td>
		<td><a href="#statistical-models">[1]</a></td>
		<td>0.78792</td>
	</tr>  
	<tr>
		<td>Decision Tree</td>
		<td></td>
		<td>TFIDF</td>
		<td><a href="#statistical-models">[1]</a></td>
		<td>0.71069</td>
	</tr>
	<tr>
		<td>XGBoost</td>
		<td></td>
		<td>TFIDF</td>
		<td><a href="#statistical-models">[1]</a></td>
		<td>(todo)</td>
	</tr>
	<tr>
		<td rowspan="2">Naive Bayes</td>
		<td>MultinomialNB</td>
		<td>TFIDF</td>
		<td><a href="#statistical-models">[1]</a></td>
		<td>0.80447</td>
	</tr>
	<tr>
		<td>ComplementNB</td>
		<td>TFIDF</td>
		<td><a href="#statistical-models">[1]</a></td>
		<td>0.79589</td>
	</tr>
</tbody>
</table>

#### Deep learning models

<table>
<thead>
    <tr>
        <th colspan="2">Model (with paper link)</th>
        <th>Pretrain parameters</th>
        <th>Training configurations</th>
        <th>Public F1</th>
        <th>Notes</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td rowspan="8"><a href="https://arxiv.org/abs/1909.11942">ALBERT</a></td>
        <td><a href="https://huggingface.co/albert-base-v1">base-v1</a></td>
        <td>11M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.80907</td>
        <td rowspan="8">View list of parameters by huggingface <a href="https://huggingface.co/transformers/v4.9.2/pretrained_models.html">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/albert-large-v1">large-v1</a></td>
        <td>17M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.80416</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/albert-xlarge-v1">xlarge-v1</a></td>
        <td>58M (huggingface)</td>
        <td><a href="#too-large-llms">[4]</a></td>
        <td>0.81182</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/albert-xxlarge-v1">xxlarge-v1</a></td>
        <td>223M (huggingface)</td>
        <td><a href="#too-large-llms">[4]</a></td>
        <td>0.78853</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/albert-base-v2">base-v2</a></td>
        <td>11M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.79528</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/albert-large-v2">large-v2</a></td>
        <td>17M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.81520</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/albert-xlarge-v2">xlarge-v2</a></td>
        <td>58M (huggingface)</td>
        <td><a href="#too-large-llms">[4]</a></td>
        <td>0.81703</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/albert-xxlarge-v2">xxlarge-v2</a></td>
        <td>223M (huggingface)</td>
        <td><a href="#too-large-llms">[4]</a></td>
        <td>0.80570</td>
    </tr>
    <tr>
        <td rowspan="4"><a href="https://arxiv.org/abs/1910.13461">BART</a></td>
        <td><a href="https://huggingface.co/facebook/bart-base">base</a></td>
        <td>140M (facebook-research)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82684</td>
        <td rowspan="4">View list of parameters by facebook-research <a href="https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/facebook/bart-large">large</a></td>
        <td>400M (facebook-research)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83726</td>
      </tr>
    <tr>
        <td><a href="https://huggingface.co/facebook/bart-large-mnli">large-mnli</a></td>
        <td>400M (facebook-research)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83450</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/facebook/bart-large-cnn">large-cnn</a></td>
        <td>400M (facebook-research)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82347</td>
    </tr>
    <tr>
        <td rowspan="8"><a href="https://arxiv.org/abs/1810.04805">BERT</a></td>
        <td><a href="https://huggingface.co/bert-base-uncased">base uncased</a></td>
        <td>110M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82899</td>
        <td rowspan="8">View list of parameters by huggingface <a href="https://huggingface.co/transformers/v4.9.2/pretrained_models.html">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/bert-base-cased">base cased</a></td>
        <td>110M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.81060</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/bert-large-uncased">large uncased</a></td>
        <td>340M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83052</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/bert-large-cased">large cased</a></td>
        <td>340M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82194</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/bert-large-uncased-whole-word-masking">large uncased whole word masking</a></td>
        <td>335M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82255</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/bert-large-cased-whole-word-masking">large cased whole word masking</a></td>
        <td>336M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.81244</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/bert-base-multilingual-uncased">multilingual uncased</a></td>
        <td>168M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.81887</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/bert-base-multilingual-cased">multilingual cased</a></td>
        <td>179M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.81918</td>
    </tr>
    <tr>
        <td rowspan="4"><a href="https://arxiv.org/abs/2005.10200">BERTweet</a></td>
        <td><a href="https://huggingface.co/vinai/bertweet-base">base</a></td>
        <td>135M (vinai)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83726</td>
        <td rowspan="4">View list of parameters by vinai <a href="https://github.com/VinAIResearch/BERTweet?tab=readme-ov-file#-pre-trained-models">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/vinai/bertweet-covid19-base-uncased">covid19-base-uncased</a></td>
        <td>135M (vinai)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.84002</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/vinai/bertweet-covid19-base-cased">covid19-base-cased</a></td>
        <td>135M (vinai)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82960</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/vinai/bertweet-large">large</a></td>
        <td>335M (vinai)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82899</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2010.10499">BORT</a></td>
        <td><a href="https://huggingface.co/amazon/bort">base</a></td>
        <td>56.1M (amazon)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.74563</td>
        <td>Parameters from the original paper</td>
    </tr>
    <tr>
        <td rowspan="4"><a href="https://arxiv.org/abs/2006.03654">DeBERTa</a></td>
        <td><a href="https://huggingface.co/microsoft/deberta-base">base</a></td>
        <td>100M (microsoft)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.81642</td>
        <td rowspan="4">View list of parameters by microsoft <a href="https://github.com/microsoft/DeBERTa?tab=readme-ov-file#pre-trained-models">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/deberta-base-mnli">base-mnli</a></td>
        <td>86M (microsoft)</td>
        <td><a href="#llms">[2]</a></td>
        <td>0.80661</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/deberta-large">large</a></td>
        <td>350M (microsoft)</td>
        <td><a href="#too-large-llms">[4]</a></td>
        <td><b>0.84308</b></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/deberta-large-mnli">large-mnli</a></td>
        <td>350M (microsoft)</td>
        <td><a href="#too-large-llms">[4]</a></td>
        <td>0.83757</td>
    </tr>
    <tr>
        <td rowspan="5"><a href="https://arxiv.org/abs/2111.09543">DeBERTa v3</a></td>
        <td><a href="https://huggingface.co/microsoft/deberta-v3-xsmall">xsmall</a></td>
        <td>22M (microsoft)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.80815</td>
        <td rowspan="5">View list of parameters by microsoft <a href="https://github.com/microsoft/DeBERTa?tab=readme-ov-file#pre-trained-models">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/deberta-v3-small">small</a></td>
        <td>44M (microsoft)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82408</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/deberta-v3-base">base</a></td>
        <td>86M (microsoft)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83205</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/deberta-v3-large">large</a></td>
        <td>304M (microsoft)</td>
        <td><a href="#too-large-llms">[4]</a></td>
        <td>0.82745</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/microsoft/mdeberta-v3-base">mdeberta-v3-base</a></td>
        <td>86M (microsoft)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82929</td>
    </tr>
    <tr>
        <td rowspan="3"><a href="https://arxiv.org/abs/1910.01108">DistilBERT</a></td>
        <td><a href="https://huggingface.co/distilbert-base-uncased">base uncased</a></td>
        <td>66M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82439</td>
        <td rowspan="3">View list of parameters by huggingface <a href="https://huggingface.co/transformers/v4.9.2/pretrained_models.html">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/distilbert-base-cased">base cased</a></td>
        <td>65M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82163</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/distilbert-base-multilingual-cased">multilingual cased</a></td>
        <td>134M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.80049</td>
    </tr>
    <tr>
        <td rowspan="3"><a href="https://arxiv.org/abs/2003.10555">ELECTRA (discriminator)</a></td>
        <td><a href="https://huggingface.co/google/electra-small-discriminator">small</a></td>
        <td>14M (google)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.81887</td>
        <td rowspan="3">View list of parameters by google <a href="https://github.com/google-research/electra">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/google/electra-base-discriminator">base</a></td>
        <td>110M (google)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82776</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/google/electra-large-discriminator">large</a></td>
        <td>335M (google)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83726</td>
    </tr>
    <tr>
        <td rowspan="4"><a href="https://arxiv.org/abs/1907.11692">RoBERTa</a></td>
        <td><a href="https://huggingface.co/roberta-base">base</a></td>
        <td>125M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82868</td>
        <td rowspan="4">View list of parameters by huggingface <a href="https://huggingface.co/transformers/v4.9.2/pretrained_models.html">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/roberta-large">large</a></td>
        <td>335M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.84033</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/roberta-large">large</a></td>
        <td>355M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.84033</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/distilroberta-base">distilroberta-large</a></td>
        <td>82M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82960</td>
    </tr>
	<tr>
		<td rowspan="3"><a href="https://arxiv.org/abs/2006.11316">SqueezeBERT</a></td>
		<td><a href="https://huggingface.co/squeezebert/squeezebert-uncased">uncased</a></td>
		<td>51M (huggingface)</td>
		<td><a href="#LLMS">[2]</a></td>
		<td>0.80324</td>
		<td rowspan="3">View list of parameters by huggingface <a href="https://huggingface.co/transformers/v4.9.2/pretrained_models.html">here</a></td>
	</tr>
    <tr>
        <td><a href="https://huggingface.co/squeezebert/squeezebert-mnli">mnli</a></td>
        <td>51M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.79987</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/squeezebert/squeezebert-mnli-headless">mnli-headless</a></td>
        <td>51M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.80416</td>
    </tr>
    <tr>
        <td rowspan="3"><a href="https://arxiv.org/abs/2010.12421">Twitter RoBERTa Sentiment</a></td>
        <td><a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment">base</a></td>
        <td>N/A</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83389</td>
        <td rowspan="3"><a href="https://huggingface.co/cardiffnlp">CardiffNLP</a> has a huge list of Twitter pretrained models and these are just 3 of them. Try finetuning others (if you have time).</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest">base latest</a></td>
        <td>N/A</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82776</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m">base 2021</a></td>
        <td>124M (cardiffnlp)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.83083</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://arxiv.org/abs/1911.02116">XLM-RoBERTa</a></td>
        <td><a href="https://huggingface.co/xlm-roberta-base">base</a></td>
        <td>270M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82439</td>
        <td rowspan="2">View list of parameters by huggingface <a href="https://huggingface.co/transformers/v4.9.2/pretrained_models.html">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/xlm-roberta-large">large</a></td>
        <td>550M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82500</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://arxiv.org/abs/1906.08237">XLNet</a></td>
        <td><a href="https://huggingface.co/xlnet-base-cased">base cased</a></td>
        <td>110M (huggingface)</td>
        <td><a href="#LLMS">[2]</a></td>
        <td>0.82592</td>
        <td rowspan="2">View list of parameters by huggingface <a href="https://huggingface.co/transformers/v4.9.2/pretrained_models.html">here</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/xlnet-large-cased">large cased</a></td>
        <td>340M (huggingface)</td>
        <td><a href="#too-large-llms">[4]</a></td>
        <td>0.81612</td>
    </tr>
    <tr>
        <td>BiLSTM Seq2Seq</td>
        <td>N/A</td>
        <td></td>
        <td><a href="#deep-learning-models">[3]</a></td>
        <td>0.78302</td>
        <td></td>
    </tr>
</tbody>
</table>