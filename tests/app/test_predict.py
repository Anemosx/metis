from unittest.mock import MagicMock, patch

import numpy as np
from fastapi import status
from fastapi.testclient import TestClient


class TestPredictDigit:

    @patch("app.routers.mnist.predict_digit")
    def test_predict_digit(
        self, mock: MagicMock, client: TestClient, create_mnist_image: tuple[str, int]
    ) -> None:

        request_image, image_label = create_mnist_image
        request_data = {"image": request_image}
        response = client.post(f"/predict-mnist", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["Content-Type"] == "application/json"

        json_response = response.json()

        assert "prediction" in json_response
        assert "distribution" in json_response

        assert isinstance(json_response["prediction"], int)
        assert isinstance(json_response["distribution"], list)

        assert 0 <= json_response["prediction"] <= 9
        assert all(isinstance(prob, float) for prob in json_response["distribution"])
        assert len(json_response["distribution"]) == 10

        assert json_response["prediction"] == image_label
        assert np.argmax(json_response["distribution"]) == json_response["prediction"]
