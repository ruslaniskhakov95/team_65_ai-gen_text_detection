from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any

class ValidationError(BaseModel):
    loc: List[Union[str, int]] = Field(title='Location')
    msg: str = Field(title='Message')
    type: str = Field(title='Error Type')

class HTTPValidationError(BaseModel):
    detail: List[ValidationError] = Field(None, title='Detail')

class LoadResponse(BaseModel):
    message: str = Field(title='Message')

    class Config:
        json_schema_extra = {
            "example":
            {
            "message": "Model 'model_name' loaded"
            }
        }

class LoadRequest(BaseModel):
    id: str = Field(title='Id')

class TransformerRequest(BaseModel):
    X: list[str]
    model_type: str

    class Config:
        json_schema={
            "example":
            {
                "X": ['As an AI language model', 'I wonder if there is a God.']
            }
        }

class UnloadRequest(BaseModel):
    id: str

class UnloadResponse(BaseModel):
    message: str = Field(title='Message')

    class Config:
        json_schema_extra = {
            "example":
            {
            "message": "Model 'model_name' unloaded"
            }
        }

class ModelListResponse(BaseModel):
    models: List[Dict[str, Any]] = Field(title='Models')

    class Config:
        json_schema_extra = {
            "example":
            {
                "models": [
                    {"id": "linear_123", "type": "linear"},
                    {"id": "linear_2", "type": "logistic"}
                ]
            }
        }

class PredictionTransformer(BaseModel):
    predictions: List[str]

    class Config:
        json_schema_extra = {
            "example":
            {
                "predictions": [
                    'machine-polished',
                    'llm'
                ]
            }
        }

class PredictionTransformerProbability(BaseModel):
    predictions: List[List[Dict[str, Union[float,str]]]]

    # class Config:
    #     json_schema_extra = {
    #         "example":
    #         {
    #             "predictions": [
    #                 'machine-polished',
    #                 'llm'
    #             ]
    #         }
    #     }