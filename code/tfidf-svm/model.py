from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC

def create_pipeline():
    ngram_range = (1, 2)
    pipeline = Pipeline([
        ("count", CountVectorizer(ngram_range=ngram_range)),
        ("tfidf", TfidfTransformer(use_idf=True)),
        ("svc", SVC(random_state=42)),
    ])

    return pipeline
