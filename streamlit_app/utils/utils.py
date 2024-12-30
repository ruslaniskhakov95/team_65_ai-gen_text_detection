import string

import aiohttp
import nltk
from nltk import tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


def tokenize_and_clean_text(text):
    tokens = tokenize.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    punct_chars = (
        string.punctuation
        + r"'s"
        + r"'t"
        + r"\n't"
        + r"'ll"
        + r"'re"
        + '""'
        + "..."
        + "'"
        + "``"
    )
    filtered_tokens = [
        word.lower()
        for word in tokens
        if word not in stop_words and word not in punct_chars and r"'" not in word
    ]
    return filtered_tokens


def lemmatize(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


predict_response = {
    0: "chatGPT",
    1: "human",
}


async def get_list_of_models():
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/list"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "error": f"Ошибка: {response.status}, {await response.text()}"
                    }
        except Exception as e:
            return {"error": f"Ошибка соединения: {str(e)}"}


async def fit_model(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/fit_corpus"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                else:
                    return response.status, {
                        "error": f"Ошибка: {response.status}, {resp}"
                    }
        except Exception as e:
            return 500, {"error": f"Ошибка соединения: {str(e)}"}


async def load_model(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/load"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                else:
                    return response.status, {
                        "error": f"Ошибка: {response.status}, {resp}"
                    }
        except Exception as e:
            return 500, {"error": f"Ошибка соединения: {str(e)}"}


async def unload_model():
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/unload"
        try:
            async with session.post(url) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                else:
                    return response.status, {
                        "error": f"Ошибка: {response.status}, {resp}"
                    }
        except Exception as e:
            return 500, {"error": f"Ошибка соединения: {str(e)}"}


async def predict_text(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/predict_text"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                else:
                    return response.status, {
                        "error": f"Ошибка: {response.status}, {resp}"
                    }
        except Exception as e:
            return 500, {"error": f"Ошибка соединения: {str(e)}"}


async def predict_corpus(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/predict_corpus"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                else:
                    return response.status, {
                        "error": f"Ошибка: {response.status}, {resp}"
                    }
        except Exception as e:
            return 500, {"error": f"Ошибка соединения: {str(e)}"}


async def preprocess_df(df):
    X = df["text"].tolist()
    y = df["label"].tolist()
    return X, y
