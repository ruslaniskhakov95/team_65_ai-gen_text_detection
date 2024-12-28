from fastapi import APIRouter, HTTPException
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from utils import (
    PredictRequest, LoadRequest, PredictMultipleRequest, PredictResponse,
    FitRequest, ApiResponse, tokenize_and_clean_text, lemmatize,
    tfidf_model, tfidf_vec, ModelListResponse
)


router = APIRouter(prefix='/api/v1/model')
models = {
    'default': [tfidf_vec, tfidf_model, 'logistic']
}
active_model = 'default'


@router.post("/fit_corpus", response_model=ApiResponse)
async def fit_new_model(request: FitRequest):
    '''Train new model'''

    global models
    config = request.config
    model_type = config.model_type.value
    vec_type = config.vec_type.value
    model_id = config.id
    params = config.hyperparams.dict()
    vec_params = config.vec_params.dict()
    X = request.X
    y = np.array(request.y)
    cleaned_texts = []
    for text in X:
        cleaned_texts.append(' '.join(
            lemmatize(tokenize_and_clean_text(text))
        ))

    if model_type == 'logistic':
        model = LogisticRegression(**params)
    elif model_type == 'svm':
        model = SVC(**params)
    if vec_type == 'bow':
        vec = CountVectorizer(**vec_params)
    elif vec_type == 'tfidf':
        vec = TfidfVectorizer(**vec_params)

    X_vec = vec.fit_transform(cleaned_texts)
    model.fit(X_vec, y)
    models[model_id] = [vec, model, model_type]

    return ApiResponse(
        message=f'Model {model_id} successfully trained! Vec: {vec_type}'
    )


@router.post("/load", response_model=ApiResponse)
async def load_model(request: LoadRequest):
    global active_model
    global models
    model_id = request.id
    if models.get(model_id):
        active_model = model_id
    else:
        raise HTTPException(
            status_code=404, detail="Requested model doesn't exist"
        )
    return ApiResponse(message=f'Model {model_id} successfully loaded')


@router.post("/unload", response_model=ApiResponse)
async def unload_model():
    global active_model
    global models
    if not active_model:
        raise HTTPException(
            status_code=404, detail="There's no active loaded model"
        )
    temp = active_model
    active_model = None
    return ApiResponse(message=f'Model {temp} unloaded!')


@router.get("/list", response_model=list[ModelListResponse])
async def list_models():
    return [
        ModelListResponse(
            id=key, type=value[2]
        ) for key, value in models.items()
    ]


@router.post("/predict_text", response_model=PredictResponse)
async def predict_model(request: PredictRequest):
    '''Prediction on a single text'''

    if not active_model:
        raise HTTPException(
            status_code=404, detail='No active model, choose one!'
        )
    text = request.X
    cleaned_text = ' '.join(lemmatize(tokenize_and_clean_text(text)))
    text_list = []
    text_list.append(cleaned_text)

    vec, model = models[active_model][0], models[active_model][1]
    text_vec = vec.transform(text_list)
    pred = model.predict(text_vec).tolist()
    return PredictResponse(prediction=pred)


@router.post("/predict_corpus", response_model=PredictResponse)
async def predict_model_corpus(request: PredictMultipleRequest):
    '''Prediction on a text corpus'''

    if not active_model:
        raise HTTPException(
            status_code=404, detail='No active model, choose one!'
        )
    corpus = request.X
    cleaned_texts = []
    for text in corpus:
        cleaned_texts.append(' '.join(
            lemmatize(tokenize_and_clean_text(text))
        ))
    vec, model = models[active_model][0], models[active_model][1]
    text_vec = vec.transform(cleaned_texts)
    pred = model.predict(text_vec).tolist()
    return PredictResponse(prediction=pred)
