import os

import torch
from PIL.Image import Image
from torch import nn
from torchvision import transforms

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


def load_mnist_model(model: nn.Module, filename: str | None = None) -> None:
    """
    Load MNIST model weights from a specified file into the provided model instance.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model instance into which the weights will be loaded.
    filename : str, optional
        The path to the file containing the model weights.
    """

    if filename is None:
        filename = os.path.join(os.getcwd(), "models", "mnist", "model.pt")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])


async def init_vision_model() -> None:
    """
    Initialize the MNIST vision model and transformations.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet().to(device)
    model.eval()
    load_mnist_model(model)

    mnist_transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    mnist_data["device"] = device
    mnist_data["model"] = model
    mnist_data["transform"] = mnist_transform


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

    image_tensor = (
        mnist_data["transform"](image)
        .unsqueeze(0)
        .to(mnist_data["device"])[:, [3], :, :]
    )

    with torch.no_grad():
        output = mnist_data["model"](image_tensor)[0]
        output_list = output.to("cpu").tolist()
        predicted_class = torch.argmax(output).item()

    return predicted_class, output_list
