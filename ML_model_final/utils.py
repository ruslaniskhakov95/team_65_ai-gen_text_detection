import nltk
import numpy as np
import string
from gensim.models import Word2Vec
from nltk import tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def tokenize_and_clean_text(text):
    tokens = tokenize.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punct_chars = string.punctuation + "'s" + '""' + '...' + "''" + '``'
    filtered_tokens = [
        word.lower() for word in tokens if word not in stop_words and word not in punct_chars
    ]
    return filtered_tokens


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [tokenize_and_clean_text(text) for text in X]


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.word2vec_model = None

    def fit(self, X, y=None):
        self.word2vec_model = Word2Vec(
            sentences=X, vector_size=self.vector_size,
            window=self.window, min_count=self.min_count,
            workers=self.workers
        )
        return self

    def transform(self, X):
        return np.array([
            np.mean([
                self.word2vec_model.wv[
                    word
                ] for word in text if word in self.word2vec_model.wv
            ] or [np.zeros(self.vector_size)], axis=0) for text in X
        ])
