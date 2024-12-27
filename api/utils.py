from enum import Enum
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from pydantic import BaseModel, ConfigDict
import string
from typing import Union

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


class AuthorPrediction(Enum):

    MACHINE = 1
    HUMAN = 0


class PredictSingleRequest(BaseModel):

    X: str


class PredictMultipleRequest(BaseModel):

    X: list[str]


class PredictResponse(BaseModel):

    prediction: Union[
        list[list[int, AuthorPrediction]], list[int, AuthorPrediction]
    ]


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


def tokenize_and_clean_text(text):
    tokens = tokenize.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punct_chars = string.punctuation + "'s" + '""' + '...' + "''" + '``'
    filtered_tokens = [
        word.lower() for word in tokens if word not in stop_words
        and word not in punct_chars
    ]
    return filtered_tokens


def lemmatize(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
