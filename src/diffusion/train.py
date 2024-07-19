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
    def __init__(self):
        super(DiffusionModel, self).__init__()
        # Encoder part
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
        # Decoder part
        self.up3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = F.relu(self.up3(x3) + x2)
        x = F.relu(self.up2(x) + x1)
        x = self.up1(x)
        return x * (0.1 + 0.9 * t.float())


def add_noise(images, t):
    noise_level = torch.sqrt(t.float()) * 0.7
    return images + torch.randn_like(images) * noise_level


def sample_time(batch_size):
    return torch.rand(batch_size, 1, 1, 1)


def train(model, train_loader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader)
        for images, _ in progress_bar:
            images = images.to(device).float()
            t = sample_time(images.size(0)).to(device)

            noisy_images = add_noise(images, t)
            optimizer.zero_grad()
            recon_images = model(noisy_images, t)
            loss = F.mse_loss(recon_images, images)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch {epoch+1} Loss {loss.item():.4f}")


def evaluate_model(model, test_loader, device, num_images=10):
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

                ax = plt.subplot(3, num_images, images_shown + 1)
                original_img = images[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
                ax.imshow(np.clip(original_img, 0, 1))
                ax.axis("off")
                ax.title.set_text("Original")

                ax = plt.subplot(3, num_images, num_images + images_shown + 1)
                noisy_img = noisy_images[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
                ax.imshow(np.clip(noisy_img, 0, 1))
                ax.axis("off")
                ax.title.set_text("Noisy")

                ax = plt.subplot(3, num_images, 2 * num_images + images_shown + 1)
                denoised_img = (
                    denoised_images[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
                )
                ax.imshow(np.clip(denoised_img, 0, 1))
                ax.axis("off")
                ax.title.set_text("Denoised")

                images_shown += 1

    plt.show()


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

    model = DiffusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, optimizer, epochs=10, device=device)
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    run()
