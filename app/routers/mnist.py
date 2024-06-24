import base64
import io
import os

from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from app.vision.mnist import predict

router = APIRouter()


class ImageData(BaseModel):
    """
    Model to represent the image data in base64 format.

    Attributes
    ----------
    image : str
        A base64-encoded string representing the image data.
    """

    image: str


mnist_html_file_path = os.path.join("static", "mnist", "index.html")
with open(mnist_html_file_path, "r") as file:
    mnist_content = file.read()


@router.get("/mnist", response_class=HTMLResponse)
async def read_mnist() -> HTMLResponse:
    """
    Serve the MNIST HTML page from the static directory.

    Returns
    -------
    response : HTMLResponse
        An HTML response containing the MNIST page content.
    """

    response = HTMLResponse(content=mnist_content)
    response.headers["Cache-Control"] = "public, max-age=3600"

    return response


@router.post("/predict-mnist")
async def predict_digit(data: ImageData) -> JSONResponse:
    """
    Decode and predict digit from base64-encoded MNIST image.

    Parameters
    ----------
    data : ImageData
        The base64-encoded image data in string format.

    Returns
    -------
    JSONResponse
        A JSON response containing the predicted digit class.
    """

    try:
        image_bytes = base64.b64decode(data.image.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        image = image.getchannel("A").resize((28, 28), Image.NEAREST)

        predicted_class, prediction_dist = predict(image)

        return JSONResponse(
            content={"prediction": predicted_class, "distribution": prediction_dist}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
