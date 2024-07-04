import torch
from torch import nn


class ConvNet(nn.Module):
    """
    A Convolutional Neural Network (ConvNet) for image classification tasks.

    Parameters
    ----------
    input_size : int, optional
        Number of input channels (default is 1).
    hidden_conv_layers : list of int, optional
        Number of filters for each hidden convolutional layer (default is [32, 64]).
    hidden_layers : list of int, optional
        Number of neurons for each hidden fully connected layer
        (default is [7 * 7 * 64, 1000]).
    num_classes : int, optional
        Number of output classes (default is 10).
    kernel_size : int, optional
        Size of the convolutional kernels (default is 5).
    stride : int, optional
        Stride for the convolutional layers (default is 1).
    padding : int, optional
        Padding for the convolutional layers (default is 2).
    max_pool_kernel_size : int, optional
        Size of the max pooling kernels (default is 2).
    max_pool_stride : int, optional
        Stride for the max pooling layers (default is 2).

    Attributes
    ----------
    conv : nn.Sequential
        Sequential container for the convolutional layers.
    drop_out : nn.Dropout
        Dropout layer to prevent overfitting.
    fc : nn.Sequential
        Sequential container for the fully connected layers.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_conv_layers: list[int] = None,
        hidden_layers: list[int] = None,
        num_classes: int = 10,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        max_pool_kernel_size: int = 2,
        max_pool_stride: int = 2,
    ):

        super(ConvNet, self).__init__()

        if hidden_conv_layers is None:
            hidden_conv_layers = [32, 64]

        if hidden_layers is None:
            hidden_layers = [7 * 7 * 64, 1000]

        conv_layers = [
            nn.Conv2d(
                input_size,
                hidden_conv_layers[0],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pool_kernel_size, stride=max_pool_stride),
        ]

        for in_channels, out_channels in zip(
            hidden_conv_layers[:-1], hidden_conv_layers[1:]
        ):
            conv_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        kernel_size=max_pool_kernel_size, stride=max_pool_stride
                    ),
                ]
            )

        self.conv = nn.Sequential(*conv_layers)

        self.drop_out = nn.Dropout()

        fc_layers = []
        for in_channels, out_channels in zip(hidden_layers[:-1], hidden_layers[1:]):
            fc_layers.extend([nn.Linear(in_channels, out_channels)])
        fc_layers.append(nn.Linear(hidden_layers[-1], num_classes))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, input_size, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes).
        """
        x = self.conv(inputs)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        out = self.fc(x)
        return out
