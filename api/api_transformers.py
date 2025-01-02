from fastapi import APIRouter, HTTPException

from util_transformers import (
    LoadResponse, TransformerRequest, HTTPValidationError, LoadRequest,
    UnloadRequest, UnloadResponse, ModelListResponse, PredictionTransformer,
    PredictionTransformerProbability, logger_setup
)
import os
from http import HTTPStatus
import traceback

from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, pipeline
)
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


path_roberta = os.getenv("ROBERTA_PATH")
path_deberta = os.getenv("DEBERTA_PATH")
path_distillberta = os.getenv("DISTILLBERTA_PATH")

model_path_map = {
    "roberta": path_roberta,
    "deberta": path_deberta,
    "distillberta": path_distillberta
}

models: dict = defaultdict(str)

router = APIRouter(prefix='/api/v1/model/transformers')

logger = logger_setup(
    name='transformers_logger', log_file='logs_transformers/log.log'
)

TEXT_CLASS_MAPPING = {
    'LABEL_2': 'Machine-Generated',
    'LABEL_0': 'Human-Written',
    'LABEL_3': 'Machine-Written, Machine-Humanized',
    'LABEL_1': 'Human-Written, Machine-Polished'
}


@router.post("/load",
             tags=['transformer detector'],
             summary='Load',
             description='Loads the a transformer model',
             responses={422: {
                'model': HTTPValidationError,
                'description': 'Validation Error'
             },
                        200: {
                            'model': LoadResponse,
                            'description': 'Successful Response'
                        }},
             status_code=HTTPStatus.OK)
async def load(request: LoadRequest):
    logger.info(f"Trying to load model with ID {request.id}")
    try:
        models[request.id] = pipeline(
            'text-classification',
            model=AutoModelForSequenceClassification.from_pretrained(
                model_path_map[request.id]
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                model_path_map[request.id]
            ),
            truncation=True,
            max_length=512,
            top_k=4
        )
        logger.info(f"Deleted model with ID {request.id}")
    except KeyError:
        logger.error(
            "Tried to load a model that does not exist: %s, \
            Here is the trace: %s",
            request.id, traceback.format_exc()
        )
        raise HTTPException(
            status_code=422,
            detail=f"Requested model doesn't exist. You can use one \
            of the following: {list(model_path_map.keys())}"
        )

    return [LoadResponse(message=f"Model '{request.id}' loaded")]


@router.post("/unload",
             tags=['transformer detector'],
             summary='Unload',
             description='Deltetes the model from the memmory',
             operation_id='unload_api_v1_models_unload_post',
             responses={
                422: {
                    'model': HTTPValidationError,
                    'description': 'Validation Error'
                },
                200: {
                    'model': UnloadResponse,
                    'description': 'Successful Response'
                }
             },
             status_code=HTTPStatus.OK)
async def unload(request: UnloadRequest):
    logger.info(f"Trying to unloadmodel with ID {request.id}")
    try:
        del models[request.id]
    except KeyError:
        logger.error(
            "Tried to unload a model that does not exist: %s, Here is the \
            trace: %s",
            request.id, traceback.format_exc()
        )
        raise HTTPException(
            status_code=422, detail=f"Requested model is not loaded."
                                    f"The currently loaded models are: "
                                    f"{list(models.keys())}"
        )

    return [UnloadResponse(message=f'unloaded model {request.id}')]


@router.get("/loaded_models",
            tags=['trainer'],
            summary='List Loaded Models',
            description='Возвращает список всех обученных моделей.',
            responses={200: {
                'model': ModelListResponse,
                'description': 'Successful Response'
            }})
async def list_models():
    logger.info('A request to list all of the currently loaded models')
    return [ModelListResponse(models=[{'models': list(models.keys())}])]


@router.get("/list_models",
            tags=['trainer'],
            summary='List All Models that are Stored on the Disk',
            description='Возвращает список всех обученных моделей.',
            responses={200: {
                'model': ModelListResponse,
                'description': 'Successful Response'
            }})
async def list_models_stored():
    logger.info('A request to list all of the models on the disk')
    return [ModelListResponse(models=[{'models': list(
        model_path_map.keys()
    )}])]


@router.post(
    "/predict", response_model=PredictionTransformer,
    status_code=HTTPStatus.OK
)
async def PredictTransformer(request: TransformerRequest):
    logger.info('Making predictions')
    try:
        predictions = models[request.model_type](request.X)
        for i, item in enumerate(predictions):
            predictions[i] = max(item, key=lambda x: x['score'])['label']
    except TypeError:
        logger.error(
            "The user tried to make a prediction with a model \
            that is not loaded or misspelled the ID (%s)! Here is the trace:\
            %s", request.id, traceback.format_exc()
        )
        raise HTTPException(
            status_code=422, detail=f"Requested model is not loaded. \
            The currently loaded models are: {list(models.keys())}"
        )
    predictions = list(map(TEXT_CLASS_MAPPING.get, predictions))
    return PredictionTransformer(predictions=predictions)


@router.post(
    "/predict_probability", response_model=PredictionTransformerProbability,
    status_code=HTTPStatus.OK
)
async def PredictTransformer(request: TransformerRequest):
    try:
        predictions = models[request.model_type](request.X)
    except TypeError:
        logger.error(
            "The user tried to make a prediction with a model \
            that is not loaded or misspelled the ID (%s)! \
            Here is the trace: %s", request.id, traceback.format_exc()
        )
        raise HTTPException(
            status_code=422,
            detail=f"Requested model is not loaded or misspelled. \
            The currently loaded models are: {list(models.keys())}"
        )

    for item in predictions[0]:
        item['label'] = TEXT_CLASS_MAPPING[item['label']]
    return PredictionTransformerProbability(predictions=predictions)
