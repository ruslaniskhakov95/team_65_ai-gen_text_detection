import json
import os
import aiohttp
from dotenv import load_dotenv

predict_response = {
    0: "chatGPT",
    1: "human",
}

load_dotenv()


API_URL = os.getenv("API_URL")


async def get_list_of_models():
    """
    Fetch the list of available models from the API.

    Returns:
        dict: A dictionary with the models or an error message.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/list"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                return {
                    "error": f"Error: {response.status}, "
                             f"{await response.text()}"
                }
        except aiohttp.ClientError as e:
            return {"error": f"HTTP error: {str(e)}"}


async def fit_model(payload):
    """
    Fit a model using the provided corpus data.

    Args:
        payload (dict): The data used to fit the model.

    Returns:
        tuple: HTTP status and the response from the API.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/fit_corpus"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                return response.status, {
                    "error": f"Error: {response.status}, {resp}"
                }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}
        except json.JSONDecodeError as e:
            return 500, {"error": f"JSON decode error: {str(e)}"}


async def load_model(payload):
    """
    Load a model using the provided payload.

    Args:
        payload (dict): The data used to load the model.

    Returns:
        tuple: HTTP status and the response from the API.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/load"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                return response.status, {
                    "error": f"Error: {response.status}, {resp}"
                }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}
        except json.JSONDecodeError as e:
            return 500, {"error": f"JSON decode error: {str(e)}"}


async def unload_model():
    """
    Unload the model from the API.

    Returns:
        tuple: HTTP status and the response from the API.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/unload"
        try:
            async with session.post(url) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                return response.status, {
                    "error": f"Error: {response.status}, {resp}"
                }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}
        except json.JSONDecodeError as e:
            return 500, {"error": f"JSON decode error: {str(e)}"}


async def predict_text(payload):
    """
    Predict text using the model.

    Args:
        payload (dict): The input data for prediction.

    Returns:
        tuple: HTTP status and the prediction result.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/predict_text"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                return response.status, {
                    "error": f"Error: {response.status}, {resp}"
                }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}
        except json.JSONDecodeError as e:
            return 500, {"error": f"JSON decode error: {str(e)}"}


async def predict_corpus(payload):
    """
    Predict using a corpus of data.

    Args:
        payload (dict): The input corpus for prediction.

    Returns:
        tuple: HTTP status and the prediction result.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/predict_corpus"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                return response.status, {
                    "error": f"Error: {response.status}, {resp}"
                }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}
        except json.JSONDecodeError as e:
            return 500, {"error": f"JSON decode error: {str(e)}"}
