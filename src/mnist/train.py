import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    patience: int = 2,
    device: str | torch.device | None = None,
    display_progress: bool = True,
) -> None:
    """
    Trains the provided model on the dataset.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    optimizer : optim.Optimizer
        The optimizer used to adjust model weights during training.
    criterion : nn.Module
        The loss function to measure the differences between predicted and true labels.
    train_loader : DataLoader
        DataLoader for the training dataset.
    epochs : int, optional
        Number of epochs to train the model (default is 5).
    patience : int, optional
        Number of epochs to wait for improvement before stopping early.
    device : str, torch.device or None, optional
        Device to run the training on.
    display_progress : bool, optional
        If True, display a progress bar with detailed training progress.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare model
    model.to(device)
    model.train()

    # initialize progress bar
    progress_bar = None
    if display_progress:
        total_steps = len(train_loader) * epochs
        progress_bar = tqdm(
            total=total_steps, desc="Training Progress", colour="white", leave=False
        )

    best_loss = float("inf")
    epochs_without_improvement = 0

    # training cycle
    for epoch in range(epochs):
        loss = 0
        correct = 0
        total = 0

        # iterate over the dataset
        for i, (images, labels) in enumerate(train_loader):

            # unpack mnist number images and labels
            images, labels = images.to(device), labels.to(device)

            # feed into the provided model
            outputs = model(images)

            # compute the differences between the predictions and true labels
            loss = criterion(outputs, labels)

            # adjust weights and optimize according to loss and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # progress display
            if display_progress:

                # calculate the accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # display the progress bar
                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{epochs} | Step {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Accuracy: {correct}/{total} ({100 * correct / total:.2f}%)"
                )
                progress_bar.update(1)

        # check for early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

        # write epoch loss to indicate some progress
        if not display_progress:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    if display_progress:
        progress_bar.close()


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: str | torch.device | None = None
) -> None:
    """
    Evaluates the provided model on a test dataset and displays results.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be evaluated.
    test_loader : DataLoader
        DataLoader for the test dataset.
    device : str, torch.device or None, optional
        Device to run the evaluation on.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare model
    model.to(device)
    model.eval()

    # prepare plotting
    plt.figure(figsize=(12, 8))
    images_to_show = 10
    images_shown = 0

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            # unpack mnist number images and labels
            images, labels = images.to(device), labels.to(device)

            # feed into the provided model
            outputs = model(images)

            # calculate the accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # plot predicted vs true labels with images
            for i in range(images.size(0)):
                if images_shown < images_to_show:

                    # plot single image
                    ax = plt.subplot(2, 5, images_shown + 1)
                    ax.axis("off")
                    ax.set_title(f"Predicted: {predicted[i]}, Actual: {labels[i]}")

                    plt.imshow(images[i].cpu().squeeze(), cmap="gray")

                    images_shown += 1
                else:
                    break

            if images_shown >= images_to_show:
                break

        # display evaluation results
        plt.show()

        # print accuracy
        print(
            f"Accuracy of the model on the {total} test images: {correct}/{total} ({100 * correct / total}%)"
        )


def save_model(model: nn.Module, filename: str = None) -> None:
    """
    Saves the state of a neural network model to a file.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be saved.
    filename : str, optional
        The path to the file where the model state should be saved.
    """

    if filename is None:
        filename = os.path.join(os.getcwd(), "models", "model.pt")

    torch.save({"model_state_dict": model.state_dict()}, filename)

    print(f"Model saved to {filename}")


def load_model(model: nn.Module, filename: str = None) -> None:
    """
    Loads the state of a neural network model from a file.

    Parameters
    ----------
    model : nn.Module
        The neural network model into which the state should be loaded.
    filename : str, optional
        The path to the file from which the model state should be loaded.
    """

    if filename is None:
        filename = os.path.join(os.getcwd(), "models", "model.pt")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded from {filename}")


def run() -> None:
    """
    Executes the full pipeline for training, saving, loading and evaluating
    a neural network model using the MNIST dataset.
    """

    # set device for the computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 20
    patience = 2

    # mnist data directory
    data_dir = os.path.join(os.getcwd(), "data")

    # image normalization
    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # train and test sets
    train_set = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=img_transform
    )
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=img_transform
    )

    # train and test loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_set.data.shape[0], shuffle=False)

    # initialize model
    model = ConvNet()

    # optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # train the model
    train_model(model, optimizer, criterion, train_loader, epochs, patience, device)

    # save weights
    save_model(model)

    # initialize and load model independently of training
    model = ConvNet()
    load_model(model)

    # evaluate the model
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    run()
