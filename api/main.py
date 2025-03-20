import os
import pickle
from contextlib import asynccontextmanager

import api_transformers
import uvicorn
from api_route import models, router
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from utils import StatusResponse


@asynccontextmanager
async def lifespan(app: FastAPI):

    global models
    current_dir = os.path.dirname(__file__)

    vec_filename = os.path.join(
        current_dir, "./baseline_OUTFOX/tfidf_vectorizer_uni.pkl"
    )
    with open(vec_filename, "rb") as vec_file:
        tfidf_vec = pickle.load(vec_file)

    model_filename = os.path.join(current_dir, "./baseline_OUTFOX/model_log_tfidf.pkl")
    with open(model_filename, "rb") as model_file:
        tfidf_model = pickle.load(model_file)

    models["default"] = [tfidf_vec, tfidf_model, "logistic"]

    yield

    models.clear()


app = FastAPI(
    lifespan=lifespan,
    title="model_inference",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

app.include_router(router)
app.include_router(api_transformers.router)


@app.exception_handler(RequestValidationError)
async def http_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.get("/", response_model=StatusResponse)
async def root() -> StatusResponse:
    return StatusResponse(status="App is online")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
