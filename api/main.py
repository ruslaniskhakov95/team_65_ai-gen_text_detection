import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import api_route
from utils import StatusResponse

app = FastAPI(
    title="model_inference",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json"
)


app.include_router(api_route.router)


@app.exception_handler(RequestValidationError)
async def http_exception_handler(
    request: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )


@app.get("/", response_model=StatusResponse)
async def root():
    return StatusResponse(status='App is online')


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
