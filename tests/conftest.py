import os
from typing import Iterable

import pytest
from fastapi.testclient import TestClient

from app.main import mnist_app

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

with open(os.path.join(data_path, "mnist_two_image.txt"), "r") as file:
    MNIST_TWO_IMAGE = file.read()


@pytest.fixture
def client() -> Iterable[TestClient]:
    with TestClient(mnist_app) as c:
        yield c


@pytest.fixture
def create_mnist_image() -> tuple[str, int]:
    return MNIST_TWO_IMAGE, 2
