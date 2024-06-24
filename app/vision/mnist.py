import os

import numpy as np
import torch
from PIL.Image import Image
from torch import nn
import torch.nn.functional as F

mnist_data = {}


class ConvNet(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.

    Methods
    -------
    forward(inputs)
        Forward pass through the network. Returns the output logits.
    """

    def __init__(self):
        """
        Initialize the Convolutional Neural Network model.

        This constructor initializes the convolutional layers, dropout layer and fully connected layers.
        """

        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the network.

        Parameters
        ----------
        inputs : torch.Tensor
            A batch of input images of shape [batch_size, 1, 28, 28], where `batch_size`
            is the number of images in the batch.

        Returns
        -------
        out : torch.Tensor
            The output logits for each class of shape [batch_size, 10].
        """

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        out = self.fc2(x)
        return out


def load_mnist_model(
    model: nn.Module,
    filename: str | None = None,
    device: str | torch.device | None = None,
) -> None:
    """
    Load MNIST model weights from a specified file into the provided model instance.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model instance into which the weights will be loaded.
    filename : str, optional
        The path to the file containing the model weights.
    device : str, torch.device or None, optional
        Device to run the evaluation on.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if filename is None:
        filename = os.path.join(os.getcwd(), "models", "mnist", "model.pt")

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


async def init_vision_model() -> None:
    """
    Initialize the MNIST vision model and transformations.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet().to(device)
    model.eval()
    load_mnist_model(model)

    mnist_data["device"] = device
    mnist_data["model"] = model


def predict(image: Image) -> tuple[int, list[float]]:
    """
    Predict the digit class for a given MNIST image.

    Parameters
    ----------
    image : PIL.Image.Image
        A PIL image representing the MNIST digit.

    Returns
    -------
    predicted_class : int
        The predicted digit class (0-9).
    output_list : list[float]
        The raw output logits for each digit class.
    """

    alpha_np = np.array(image, dtype=np.float32) / 255.0

    image_tensor = (
        torch.tensor(alpha_np).unsqueeze(0).unsqueeze(0).to(mnist_data["device"])
    )
    image_tensor = (image_tensor - 0.5) * 2

    with torch.no_grad():
        output = mnist_data["model"](image_tensor)[0]
        output_list = F.softmax(output, dim=0).to("cpu").tolist()
        predicted_class = torch.argmax(output).item()

    return predicted_class, output_list
