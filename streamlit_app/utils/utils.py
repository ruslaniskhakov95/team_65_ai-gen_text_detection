import aiohttp

predict_response = {
    0: "chatGPT",
    1: "human",
}


async def get_list_of_models():
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/list"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "error": f"Ошибка: {response.status}, "
                                 f"{await response.text()}"
                    }
        except Exception as e:
            return {"error": f"Ошибка соединения: {str(e)}"}


async def fit_model(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/fit_corpus"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                print(resp)
                if response.status == 200:
                    return response.status, resp
                else:
                    return response.status, {
                        "error": f"Ошибка: {response.status}, {resp}"
                    }
        except Exception as e:
            return 500, {"error": f"Ошибка соединения: {str(e)}"}


async def load_model(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/load"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                else:
                    return response.status, {
                        "error": f"Ошибка: {response.status}, {resp}"
                    }
        except Exception as e:
            return 500, {"error": f"Ошибка соединения: {str(e)}"}


async def unload_model():
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/unload"
        try:
            async with session.post(url) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                else:
                    return response.status, {
                        "error": f"Ошибка: {response.status}, {resp}"
                    }
        except Exception as e:
            return 500, {"error": f"Ошибка соединения: {str(e)}"}


async def predict_text(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/predict_text"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                else:
                    return response.status, {
                        "error": f"Ошибка: {response.status}, {resp}"
                    }
        except Exception as e:
            return 500, {"error": f"Ошибка соединения: {str(e)}"}


async def predict_corpus(payload):
    async with aiohttp.ClientSession() as session:
        url = "http://127.0.0.1:8000/api/v1/model/predict_corpus"
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.json()
                if response.status == 200:
                    return response.status, resp
                else:
                    return response.status, {
                        "error": f"Ошибка: {response.status}, {resp}"
                    }
        except Exception as e:
            return 500, {"error": f"Ошибка соединения: {str(e)}"}
