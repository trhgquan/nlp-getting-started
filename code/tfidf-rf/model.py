from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

def create_pipeline():
    ngram_range = (1, 1)
    pipeline = Pipeline([
        ("count", CountVectorizer(ngram_range=ngram_range)),
        ("tfidf", TfidfTransformer(use_idf=True)),
        ("svc", RandomForestClassifier(random_state=42)),
    ])

    return pipeline
