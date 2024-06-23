import base64
import io

from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
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

        predicted_class, prediction_dist = predict(image)

        return JSONResponse(content={"prediction": predicted_class})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
