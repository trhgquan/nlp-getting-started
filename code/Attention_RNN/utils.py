import re
import string

# Clear emojis


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Clear html


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

# Clear urls


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

# Clear special characters


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def clean_df(df):
    # Lowering
    df["text"] = df["text"].apply(lambda x: x.lower())

    # Apply text cleaning
    df["text"] = df["text"].apply(lambda x: remove_emoji(x))
    df["text"] = df["text"].apply(lambda x: remove_html(x))
    df["text"] = df["text"].apply(lambda x: remove_URL(x))
    df["text"] = df["text"].apply(lambda x: remove_punct(x))

    # Remove multiple spaces
    df["text"] = df.text.replace("\s+", " ", regex=True)

    return df


def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = remove_emoji(sentence)
    sentence = remove_html(sentence)
    sentence = remove_punct(sentence)
    sentence = remove_URL(sentence)
    sentence = sentence.replace("\s+", " ", regex=True)

    return sentence
