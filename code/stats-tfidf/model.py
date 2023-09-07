from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC


def create_pipeline(classifier=SVC):
    ngram_range = (1, 2)
    pipeline = Pipeline([
        ("count", CountVectorizer(ngram_range=ngram_range)),
        ("tfidf", TfidfTransformer(use_idf=True)),
        ("clf", classifier(random_state=42)),
    ])

    return pipeline
