import json
import os
import aiohttp
from dotenv import load_dotenv

label_map = {
    0: "chatGPT",
    1: "human",
    2: "cohere",
    3: "davinci",
    4: "bloomz",
    5: "dolly",
    6: "gpt-4",
}


load_dotenv()


API_URL = os.getenv("API_URL")


async def load_model(payload):
    """
        Load a model using the provided payload.

        Args:
            payload (dict): The data used to load the model.

        Returns:
            tuple: the response from the API.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/transformers/load"
        try:
            async with session.post(url, json={'id': payload}) as response:
                if response.status == 200:
                    resp = await response.json()
                    return list(resp)
                return {
                        "error": f"Error: {response.status}, "
                                 f"{await response.text()}"
                    }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}
        except json.JSONDecodeError as e:
            return 500, {"error": f"JSON decode error: {str(e)}"}


async def unload_model(payload):
    """
        Unload the model from the API.

        Args:
            payload (dict): The data used to unload the model.

        Returns:
            tuple: the response from the API.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/transformers/unload"
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    resp = await response.json()
                    return list(resp)
                return {
                        "error": f"Error: {response.status}, "
                                 f"{await response.text()}"
                    }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}
        except json.JSONDecodeError as e:
            return 500, {"error": f"JSON decode error: {str(e)}"}


async def get_list_of_models():
    """
        Fetch the list of available models from the API.

        Returns:
            dict: A dictionary with the models or an error message.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/transformers/list_models"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    resp = await response.json()
                    return resp[0]["models"][0]["models"]
                return {
                        "error": f"Ошибка: {response.status}, "
                                 f"{await response.text()}"
                    }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}


async def get_list_of_loaded_models():
    """
        Fetch the list of loaded models from the API.

        Returns:
            dict: A dictionary with the models or an error message.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/transformers/loaded_models"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    resp = await response.json()
                    return resp[0]["models"][0]["models"]
                return {
                        "error": f"Ошибка: {response.status}, "
                                 f"{await response.text()}"
                    }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}


async def predict(payload):
    """
        Predict using a corpus of data.

        Args:
            payload (dict): The input corpus for prediction.

        Returns:
            tuple: the prediction result.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/transformers/predict"
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    resp = await response.json()
                    return resp
                return {
                        "error": f"Ошибка: {response.status}, "
                                 f"{await response.text()}"
                    }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}
        except json.JSONDecodeError as e:
            return 500, {"error": f"JSON decode error: {str(e)}"}


async def predict_probability(payload):
    """
        Predict using a corpus of data.

        Args:
            payload (dict): The input corpus for prediction.

        Returns:
            tuple: the prediction result.
    """
    async with aiohttp.ClientSession() as session:
        url = API_URL + "/api/v1/model/transformers/predict_probability"
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    resp = await response.json()
                    return resp
                return {
                        "error": f"Ошибка: {response.status}, "
                                 f"{await response.text()}"
                    }
        except aiohttp.ClientError as e:
            return 500, {"error": f"HTTP error: {str(e)}"}
        except json.JSONDecodeError as e:
            return 500, {"error": f"JSON decode error: {str(e)}"}
