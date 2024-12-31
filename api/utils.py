from enum import Enum
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
import os
import pickle
from pydantic import BaseModel, ConfigDict
import string
from typing import List, Optional

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


current_dir = os.path.dirname(__file__)
vec_filename = os.path.join(
    current_dir, '../baseline_OUTFOX/tfidf_vectorizer_uni.pkl'
)
with open(vec_filename, 'rb') as vec_file:
    tfidf_vec = pickle.load(vec_file)


model_filename = os.path.join(
    current_dir, '../baseline_OUTFOX/model_log_tfidf.pkl'
)
with open(model_filename, 'rb') as model_file:
    tfidf_model = pickle.load(model_file)


class AuthorPrediction(Enum):

    MACHINE = 1
    HUMAN = 0


class ModelType(Enum):

    logistic = 'logistic'
    svm = 'svm'


class VectorizerType(Enum):

    bow = 'bow'
    tfidf = 'tfidf'


class HyperParams(BaseModel):
    C: float = 1.0
    fit_intercept: bool = False
    random_state: int = None
    verbose: int = 0


class VecParams(BaseModel):
    max_features: Optional[int] = None


class LoadRequest(BaseModel):

    id: str


class ModelConfig(LoadRequest):
    hyperparams: HyperParams
    model_type: ModelType
    vec_type: VectorizerType
    vec_params: VecParams


class PredictRequest(BaseModel):

    X: str


class PredictMultipleRequest(PredictRequest):

    X: List[str]


class FitRequest(PredictMultipleRequest):

    config: ModelConfig
    y: List[int]


class ApiResponse(BaseModel):

    message: str


class FitResponse(ApiResponse):

    train_sizes: List
    train_scores_mean: List
    test_scores_mean: List
    train_scores_std: List
    test_scores_std: List


class PredictResponse(BaseModel):

    probability: list[list[float]]


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


class ModelListResponse(BaseModel):
    id: str
    type: str


def tokenize_and_clean_text(text):
    tokens = tokenize.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punct_chars = string.punctuation + r"'s" + r"'t" + r"\n't" + r"'ll" + r"'re" + '""' + '...' + "\'" + '``'
    filtered_tokens = [
        word.lower() for word in tokens if word not in stop_words and word not in punct_chars and r"'" not in word
    ]
    return filtered_tokens


def lemmatize(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
