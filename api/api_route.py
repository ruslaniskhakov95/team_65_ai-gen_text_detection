from fastapi import APIRouter
import pickle
import os

from utils import (
    PredictSingleRequest, PredictMultipleRequest, PredictResponse,
    tokenize_and_clean_text, lemmatize
)


router = APIRouter(prefix='/api/v1/model')


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


@router.post("/predict_text", response_model=PredictResponse)
async def predict_model(request: PredictSingleRequest):
    '''Prediction on a single text'''

    global tfidf_vec
    global tfidf_model
    text = request.X
    cleaned_text = ' '.join(lemmatize(tokenize_and_clean_text(text)))
    text_list = []
    text_list.append(cleaned_text)
    text_vec = tfidf_vec.transform(text_list)
    pred = tfidf_model.predict(text_vec).tolist()
    print(tfidf_model.predict_proba(text_vec))
    return PredictResponse(prediction=pred)


@router.post("/predict_corpus", response_model=PredictResponse)
async def predict_model_corpus(request: PredictMultipleRequest):
    '''Prediction on a text corpus'''

    global tfidf_vec
    global tfidf_model
    corpus = request.X
    cleaned_texts = []
    for text in corpus:
        cleaned_texts.append(' '.join(
            lemmatize(tokenize_and_clean_text(text))
        ))
    text_vec = tfidf_vec.transform(cleaned_texts)
    pred = tfidf_model.predict(text_vec).tolist()
    return PredictResponse(prediction=pred)
