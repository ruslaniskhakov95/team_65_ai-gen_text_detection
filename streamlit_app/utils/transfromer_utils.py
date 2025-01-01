import aiohttp

label_map = {
    0: "chatGPT",
    1: "human",
    2: "cohere",
    3: "davinci",
    4: "bloomz",
    5: "dolly",
    6: "gpt-4",
}


async def load_model(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/transformers/load"
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    resp = await response.json()
                    return [r for r in resp]
                else:
                    return {
                        "error": f"Ошибка: {response.status}, "
                                 f"{await response.text()}"
                    }
        except Exception as e:
            return {"error": f"Ошибка соединения: {str(e)}"}


async def unload_model(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/transformers/unload"
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    resp = await response.json()
                    return [r for r in resp]
                else:
                    return {
                        "error": f"Ошибка: {response.status}, "
                                 f"{await response.text()}"
                    }
        except Exception as e:
            return {"error": f"Ошибка соединения: {str(e)}"}


async def get_list_of_models():
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/transformers/list_models"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    resp = await response.json()
                    return resp[0]["models"][0]["models"]
                else:
                    return {
                        "error": f"Ошибка: {response.status}, "
                                 f"{await response.text()}"
                    }
        except Exception as e:
            return {"error": f"Ошибка соединения: {str(e)}"}


async def get_list_of_loaded_models():
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/transformers/loaded_models"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    resp = await response.json()
                    return resp[0]["models"][0]["models"]
                else:
                    return {
                        "error": f"Ошибка: {response.status},"
                                 f" {await response.text()}"
                    }
        except Exception as e:
            return {"error": f"Ошибка соединения: {str(e)}"}


async def predict(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/transformers/predict"
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    resp = await response.json()
                    return [r for r in resp]
                else:
                    return {
                        "error": f"Ошибка: {response.status}, "
                                 f"{await response.text()}"
                    }
        except Exception as e:
            return {"error": f"Ошибка соединения: {str(e)}"}
