import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class DiffusionModel(nn.Module):
    """
    A simple neural network model for diffusion-based image denoising.

    Methods
    -------
    forward(x, t)
        Forward pass through the network. Applies the diffusion model to input images.
    """

    def __init__(self):
        """
        Initialize the DiffusionModel with convolutional layers.

        The model uses three downsampling convolutional layers and three upsampling layers to denoise images.
        """

        super(DiffusionModel, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.up3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, 3, height, width].
        t : torch.Tensor
            Time step tensor used to modulate the output.

        Returns
        -------
        torch.Tensor
            The denoised output tensor of shape [batch_size, 3, height, width].
        """

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x = F.relu(self.up3(x3) + x2)
        x = F.relu(self.up2(x) + x1)
        x = self.up1(x)

        return x * (0.1 + 0.9 * t.float())


def add_noise(images: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Adds noise to images based on the time step `t`.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images to add noise to, of shape [batch_size, 3, height, width].
    t : torch.Tensor
        Time step tensor that determines the amount of noise to add.

    Returns
    -------
    torch.Tensor
        Noisy images with the same shape as the input images.
    """

    noise_level = torch.sqrt(t.float()) * 0.7

    # add Gaussian noise
    return images + torch.randn_like(images) * noise_level


def sample_time(batch_size: int) -> torch.Tensor:
    """
    Samples a random time step for each image in a batch.

    Parameters
    ----------
    batch_size : int
        The number of images in the batch.

    Returns
    -------
    torch.Tensor
        Random time steps of shape [batch_size, 1, 1, 1].
    """

    return torch.rand(batch_size, 1, 1, 1)


def train(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
          epochs: int, device: torch.device) -> None:
    """
    Trains the diffusion model on a given dataset.

    Parameters
    ----------
    model : nn.Module
        The diffusion model to be trained.
    train_loader : DataLoader
        DataLoader for the training dataset.
    optimizer : optim.Optimizer
        Optimizer used to adjust model weights during training.
    epochs : int
        Number of epochs to train the model.
    device : torch.device
        Device to run the training on (CPU or CUDA).
    """

    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader)
        for images, _ in progress_bar:
            images = images.to(device).float()
            t = sample_time(images.size(0)).to(device)

            # add noise for training
            noisy_images = add_noise(images, t)

            optimizer.zero_grad()

            # reconstruct the image
            recon_images = model(noisy_images, t)

            # calculate the loss
            loss = F.mse_loss(recon_images, images)

            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch {epoch + 1} Loss {loss.item():.4f}")


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device,
                   num_images: int = 10) -> None:
    """
    Evaluates the model on a test dataset and visualizes the results.

    Parameters
    ----------
    model : nn.Module
        The diffusion model to be evaluated.
    test_loader : DataLoader
        DataLoader for the test dataset.
    device : torch.device
        Device to run the evaluation on (CPU or CUDA).
    num_images : int, optional
        Number of images to visualize (default is 10).
    """

    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 5))

    with torch.no_grad():
        for images, _ in test_loader:
            if images_shown >= num_images:
                break
            images = images.to(device).float()

            t = sample_time(images.size(0)).to(device)
            noisy_images = add_noise(images, t)

            denoised_images = model(noisy_images, t)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break

                # original image
                ax = plt.subplot(3, num_images, images_shown + 1)
                original_img = images[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
                ax.imshow(np.clip(original_img, 0, 1))
                ax.axis("off")
                ax.title.set_text("Original")

                # noisy image
                ax = plt.subplot(3, num_images, num_images + images_shown + 1)
                noisy_img = noisy_images[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
                ax.imshow(np.clip(noisy_img, 0, 1))
                ax.axis("off")
                ax.title.set_text("Noisy")

                # denoised image
                ax = plt.subplot(3, num_images, 2 * num_images + images_shown + 1)
                denoised_img = (
                        denoised_images[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
                )
                ax.imshow(np.clip(denoised_img, 0, 1))
                ax.axis("off")
                ax.title.set_text("Denoised")

                images_shown += 1

    plt.show()


def run() -> None:
    """
    Runs the full pipeline for training and evaluating the diffusion model using the CIFAR-10 dataset.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

    # initialize model and optimizer
    model = DiffusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train the model
    train(model, train_loader, optimizer, epochs=10, device=device)

    # evaluate the model
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    run()
