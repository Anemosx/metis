import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import base_router, predict_router
from app.vision.mnist import init_vision_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes FastAPI application.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application.
    """

    await init_model()
    yield


mnist_app: FastAPI = FastAPI(
    title="Vision Service",
    version=os.getenv("VERSION", "0.0.1"),
    lifespan=lifespan,
)
mnist_app.mount("/static", StaticFiles(directory="static"), name="static")
mnist_app.include_router(base_router)
mnist_app.include_router(predict_router)


async def init_model() -> None:
    """
    Initializes machine learning model.
    """
    await init_vision_model()


if __name__ == "__main__":
    uvicorn.run(mnist_app, port=8000)
