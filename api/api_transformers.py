import os
import traceback
from collections import defaultdict
from http import HTTPStatus

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from util_transformers import (HTTPValidationError, LoadRequest, LoadResponse,
                               ModelListResponse, PredictionTransformer,
                               PredictionTransformerProbability,
                               TransformerRequest, UnloadRequest,
                               UnloadResponse, logger_setup)

# from api_route import models

load_dotenv()


path_roberta = os.getenv("ROBERTA_PATH")
path_deberta = os.getenv("DEBERTA_PATH")
path_distillberta = os.getenv("DISTILLBERTA_PATH")

model_path_map = {
    "roberta": path_roberta,
    "deberta": path_deberta,
    "distillberta": path_distillberta,
}

models: dict = defaultdict(str)

router = APIRouter(prefix="/api/v1/model/transformers")

current_dir = os.path.dirname(__file__)
logger = logger_setup(
    name="transformers_logger",
    log_file=os.path.join(current_dir, "./logs_transformers/log.log"),
)

TEXT_CLASS_MAPPING = {
    "LABEL_2": "Machine-Generated",
    "LABEL_0": "Human-Written",
    "LABEL_3": "Machine-Written, Machine-Humanized",
    "LABEL_1": "Human-Written, Machine-Polished",
}


@router.post(
    "/load",
    tags=["transformer detector"],
    summary="Load",
    description="Loads the a transformer model",
    responses={
        422: {"model": HTTPValidationError, "description": "Validation Error"},
        200: {"model": LoadResponse, "description": "Successful Response"},
    },
    status_code=HTTPStatus.OK,
)
async def load(request: LoadRequest):
    """Load transformer model into memory"""
    logger.info(f"Trying to load model with ID {request.id}")
    try:
        models[request.id] = [
            pipeline(
                "text-classification",
                model=AutoModelForSequenceClassification.from_pretrained(
                    model_path_map[request.id]
                ),
                tokenizer=AutoTokenizer.from_pretrained(model_path_map[request.id]),
                truncation=True,
                max_length=512,
                top_k=4,
            ),
            "auto",
            request.id,
        ]
        logger.info(f"Deleted model with ID {request.id}")
    except KeyError:
        logger.error(
            "Tried to load a model that does not exist: %s, \
            Here is the trace: %s",
            request.id,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=422,
            detail=f"Requested model doesn't exist. You can use one \
            of the following: {list(model_path_map.keys())}",
        )

    return [LoadResponse(message=f"Model '{request.id}' loaded")]


@router.post(
    "/unload",
    tags=["transformer detector"],
    summary="Unload",
    description="Deletes the model from the memory",
    operation_id="unload_api_v1_models_unload_post",
    responses={
        422: {"model": HTTPValidationError, "description": "Validation Error"},
        200: {"model": UnloadResponse, "description": "Successful Response"},
    },
    status_code=HTTPStatus.OK,
)
async def unload(request: UnloadRequest):
    """Unload transformer model from memory"""
    logger.info(f"Trying to unload model with ID {request.id}")
    try:
        del models[request.id]
    except KeyError:
        logger.error(
            "Tried to unload a model that does not exist: %s, Here is the \
            trace: %s",
            request.id,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=422,
            detail=f"Requested model is not loaded."
            f"The currently loaded models are: "
            f"{list(models.keys())}",
        )

    return [UnloadResponse(message=f"unloaded model {request.id}")]


@router.get(
    "/loaded_models",
    tags=["loader"],
    summary="List Loaded Models",
    description="List Loaded Models",
    responses={200: {"model": ModelListResponse, "description": "Successful Response"}},
)
async def list_models():
    """List Loaded Models"""
    logger.info("A request to list all of the currently loaded models")
    return [ModelListResponse(models=[{"models": list(models.keys())}])]


@router.get(
    "/list_models",
    tags=["trainer"],
    summary="List of all available models",
    description="List All Models that are Stored on the Disk",
    responses={200: {"model": ModelListResponse, "description": "Successful Response"}},
)
async def list_models_stored():
    """List Transformers Models"""
    logger.info("A request to list all of the models on the disk")
    return [ModelListResponse(models=[{"models": list(model_path_map.keys())}])]


@router.post(
    "/predict",
    tags=["predict_text"],
    summary="Predict a text",
    description="Predict text with available model",
    responses={
        422: {"model": HTTPValidationError, "description": "Validation Error"},
        200: {"model": PredictionTransformer, "description": "Successful Response"},
    },
    status_code=HTTPStatus.OK,
)
async def PredictTransformer(request: TransformerRequest):
    """Predict Label of Text"""
    logger.info("Making predictions")
    try:
        predictions = models[request.model_type](request.X)
        for i, item in enumerate(predictions):
            predictions[i] = max(item, key=lambda x: x["score"])["label"]
    except TypeError:
        logger.error(
            "The user tried to make a prediction with a model \
            that is not loaded or misspelled the ID (%s)! Here is the trace:\
            %s",
            request.id,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=422,
            detail=f"Requested model is not loaded. \
            The currently loaded models are: {list(models.keys())}",
        )
    predictions = list(map(TEXT_CLASS_MAPPING.get, predictions))
    return PredictionTransformer(predictions=predictions)


@router.post(
    "/predict_probability",
    response_model=PredictionTransformerProbability,
    tags=["pred_proba"],
    summary="Predict proba",
    description="Predict probability of text label",
    responses={
        422: {"model": HTTPValidationError, "description": "Validation Error"},
        200: {
            "model": PredictionTransformerProbability,
            "description": "Successful Response",
        },
    },
    status_code=HTTPStatus.OK,
)
async def PredictProbaTransformer(request: TransformerRequest):
    """Predict Probability of Text"""
    try:
        predictions = models[request.model_type](request.X)
    except TypeError:
        logger.error(
            "The user tried to make a prediction with a model \
            that is not loaded or misspelled the ID (%s)! \
            Here is the trace: %s",
            request.id,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=422,
            detail=f"Requested model is not loaded or misspelled. \
            The currently loaded models are: {list(models.keys())}",
        )

    for item in predictions[0]:
        item["label"] = TEXT_CLASS_MAPPING[item["label"]]
    return PredictionTransformerProbability(predictions=predictions)
