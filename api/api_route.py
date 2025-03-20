import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus

import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from util_transformers import HTTPValidationError, logger_setup
from utils import (ApiResponse, FitRequest, FitResponse, LoadRequest,
                   ModelListResponse, PredictMultipleRequest, PredictRequest,
                   PredictResponse, lemmatize, tokenize_and_clean_text)

models = {}
active_model = "default"


router = APIRouter(prefix="/api/v1/model")
executor = ThreadPoolExecutor(max_workers=1)

current_dir = os.path.dirname(__file__)
logger = logger_setup(
    name="basic_logger", log_file=os.path.join(current_dir, "./logs_basic/log.log")
)


@router.post(
    "/fit_corpus",
    tags=["fit"],
    summary="Train a new model",
    description="This endpoint trains a new model using the provided corpus data and configuration.",
    responses={
        422: {"model": HTTPValidationError, "description": "Validation Error"},
        200: {"model": FitResponse, "description": "Successful Response"},
    },
    status_code=HTTPStatus.OK,
)
async def fit_new_model(request: FitRequest):
    """Train new model"""

    global models
    config = request.config
    model_type = config.model_type.value
    vec_type = config.vec_type.value
    model_id = config.id
    params = config.hyperparams.dict()
    vec_params = config.vec_params.dict()
    X = request.X
    y = np.array(request.y)
    logger.info(f"Trying to fit model of type {model_type} and vectorizer {vec_type}")

    if model_type == "logistic":
        model = LogisticRegression(**params)
    elif model_type == "svm":
        model = SVC(**params)
    if vec_type == "bow":
        vec = CountVectorizer(**vec_params)
    elif vec_type == "tfidf":
        vec = TfidfVectorizer(**vec_params)

    X_vec = vec.fit_transform(X)

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_vec,
        y,
        n_jobs=-1,
        train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    mean_train_scores = train_scores.mean(axis=1)
    mean_test_scores = test_scores.mean(axis=1)
    std_train_scores = train_scores.std(axis=1)
    std_test_scores = test_scores.std(axis=1)

    def train() -> None:
        model.fit(X_vec, y)
        models[model_id] = [vec, model, model_type]

    try:
        future = executor.submit(train)
        future.result(timeout=10)
        logger.info(f"Successfully trained model with id {model_id}")
    except TimeoutError:
        logger.error(
            "Tried to fit model with id: %s, but got a Timeout error.\
            Here is the trace: %s",
            model_id,
            traceback.format_exc(),
        )
        raise HTTPException(status_code=408, detail="Train timeout")

    return FitResponse(
        message=f"Model {model_id} successfully trained! Vec: {vec_type}",
        train_sizes=train_sizes.tolist(),
        train_scores_mean=mean_train_scores.tolist(),
        test_scores_mean=mean_test_scores.tolist(),
        train_scores_std=std_train_scores.tolist(),
        test_scores_std=std_test_scores.tolist(),
    )


@router.post(
    "/load",
    tags=["load"],
    summary="Load a model",
    description="Load a model to disk from fitting models",
    responses={
        422: {"model": HTTPValidationError, "description": "Validation Error"},
        200: {"model": ApiResponse, "description": "Successful Response"},
    },
    status_code=HTTPStatus.OK,
)
async def load_model(request: LoadRequest):
    """Loads model"""

    global active_model
    global models
    model_id = request.id
    logger.info("Trying to load model with id: %s", model_id)
    if models.get(model_id):
        active_model = model_id
    else:
        logger.error(
            "Tried to load a model that does not exist: %s, \
            Here is the trace: %s",
            request.id,
            traceback.format_exc(),
        )
        raise HTTPException(status_code=404, detail="Requested model doesn't exist")
    logger.info("Loaded model with id: %s", model_id)
    return ApiResponse(message=f"Model {model_id} successfully loaded")


@router.post(
    "/unload",
    tags=["unload"],
    summary="Unload a model",
    description="Unload a model from inference",
    responses={
        422: {"model": HTTPValidationError, "description": "Validation Error"},
        200: {"model": ApiResponse, "description": "Successful Response"},
    },
    status_code=HTTPStatus.OK,
)
async def unload_model():
    """Unloads model from inference"""

    global active_model
    global models
    logger.info("Trying to unload the current active model")
    if not active_model:
        logger.error(
            "There is no active model! Here is the traceback: %s",
            traceback.format_exc(),
        )
        raise HTTPException(status_code=404, detail="There's no active loaded model")
    logger.info("Successfully unloaded the active model")
    temp = active_model
    active_model = None
    return ApiResponse(message=f"Model {temp} unloaded!")


@router.get(
    "/list",
    tags=["models_list"],
    summary="Show available models",
    description="Show available fitting models",
    responses={200: {"model": ModelListResponse, "description": "Successful Response"}},
    status_code=HTTPStatus.OK,
)
async def list_models():
    """List all available models"""

    logger.info("A request to list all of the models on the disk")
    return [ModelListResponse(id=key, type=value[2]) for key, value in models.items()]


@router.post(
    "/predict_text",
    tags=["predict_text"],
    summary="Predict a text",
    description="Predict solo text with available model",
    responses={
        422: {"model": HTTPValidationError, "description": "Validation Error"},
        200: {"model": PredictResponse, "description": "Successful Response"},
    },
    status_code=HTTPStatus.OK,
)
async def predict_model(request: PredictRequest):
    """Prediction on a single text"""

    logger.info("Trying to make a prediction with the current active model.")
    if not active_model:
        logger.error(
            "There is no active model! Here is the traceback: %s",
            traceback.format_exc(),
        )
        raise HTTPException(status_code=404, detail="No active model, choose one!")
    text = request.X
    cleaned_text = " ".join(lemmatize(tokenize_and_clean_text(text)))
    text_list = []
    text_list.append(cleaned_text)

    vec, model = models[active_model][0], models[active_model][1]
    text_vec = vec.transform(text_list)
    pred = model.predict_proba(text_vec).tolist()
    logger.info("Successfully made the prediction.")
    return PredictResponse(probability=pred)


@router.post(
    "/predict_corpus",
    tags=["predict_texts"],
    summary="Predict a text corpus",
    description="Predict text corpus with available model",
    responses={
        422: {"model": HTTPValidationError, "description": "Validation Error"},
        200: {"model": PredictResponse, "description": "Successful Response"},
    },
    status_code=HTTPStatus.OK,
)
async def predict_model_corpus(request: PredictMultipleRequest):
    """Prediction on a text corpus"""

    logger.info("Trying to make a corpus prediction with the current active model.")
    if not active_model:
        logger.error(
            "There is no active model! Here is the traceback: %s",
            traceback.format_exc(),
        )
        raise HTTPException(status_code=404, detail="No active model, choose one!")
    corpus = request.X
    cleaned_texts = []
    for text in corpus:
        cleaned_texts.append(" ".join(lemmatize(tokenize_and_clean_text(text))))
    vec, model = models[active_model][0], models[active_model][1]
    text_vec = vec.transform(cleaned_texts)
    pred = model.predict_proba(text_vec).tolist()
    logger.info("Successfully made the corpus prediction.")
    return PredictResponse(probability=pred)
