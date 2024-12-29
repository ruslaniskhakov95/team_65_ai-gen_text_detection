from fastapi import APIRouter

from util_transformers import (LoadResponse, TransformerRequest, HTTPValidationError, 
                            LoadRequest, UnloadRequest, UnloadResponse, ModelListResponse, PredictionTransformer)
import os
from http import HTTPStatus

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


path_roberta = os.getenv("ROBERTA_PATH")
path_deberta = os.getenv("DEBERTA_PATH")
path_distillberta = os.getenv("DISTILLBERTA_PATH")

model_path_map = {
    "roberta":path_roberta,
    "deberta":path_deberta,
    "distillberta":path_distillberta
}

models: dict = defaultdict(str)

router = APIRouter(prefix='/api/v1/model/transformers')

@router.post("/load",
             tags=['transformer detector'],
             summary= 'Load',
             description= 'Loads the a transformer model',
             responses = {422: {'model': HTTPValidationError, 'description': 'Validation Error'}, 
                          200: {'model': LoadResponse, 'description': 'Successful Response'}},
             status_code=HTTPStatus.OK)
async def load(request:LoadRequest):
    models[request.id] = pipeline('text-classification', 
                                  model= AutoModelForSequenceClassification.from_pretrained(model_path_map[request.id]), 
                                  tokenizer=AutoTokenizer.from_pretrained(model_path_map[request.id]), 
                                  truncation=True, 
                                  max_length=512, 
                                  top_k=4)
    return [LoadResponse(message=f"Model '{request.id}' loaded")]

@router.post("/unload",
             tags=['transformer detector'],
             summary= 'Unload',
             description= 'Deltetes the model from the memmory',
             operation_id='unload_api_v1_models_unload_post',
             responses = {422: {'model': HTTPValidationError, 'description': 'Validation Error'}, 
                          200: {'model': UnloadResponse, 'description': 'Successful Response'}},
             status_code=HTTPStatus.OK)
async def unload(request:UnloadRequest):
    del models[request.id]
    return [UnloadResponse(message=f'unloaded model {request.id}')]

@router.get("/list_models",
             tags=['trainer'],
             summary= 'List Models',
             description= 'Возвращает список всех обученных моделей.',
             operation_id='list_models_api_v1_models_list_models_get',
             responses = {200: {'model': ModelListResponse, 'description': 'Successful Response'}})
async def list_models():
    return [ModelListResponse(models=[{'models':list(models.keys())}])]


@router.post("/predict", response_model=PredictionTransformer, status_code=HTTPStatus.OK)
async def PredictTransformer(request: TransformerRequest):
    predictions = models[request.model_type](request.X)
    for i, item in enumerate(predictions):
        predictions[i] = max(item, key=lambda x: x['score'])['label']
    return PredictionTransformer(predictions=predictions)