import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
    """
    Encoder for a Variational Autoencoder (VAE) for an OGM
    """

    def __init__(self, dimensions: tuple, embedding_dim, debug=False):
        """
        Initializes the CNN encoder

        Parameters
        ----------
        dimensions : tuple
            Dimensions of the input grid (x_dim, y_dim)
        """
        super(VariationalEncoder, self).__init__()
        self.dimensions = dimensions
        self.debug = debug

        # Define the CNN layers
        self.conv1 = nn.Conv2d(
            1, 8, kernel_size=5, stride=1, padding=1
        )  # Gets preppadded with ones

        self.prelu1 = nn.PReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU()

        # flattened_size = 32 * (dimensions[0] - 6) * (dimensions[1] - 6)
        flattened_size = 3200
        print("flattened size: ", flattened_size)

        self.fc1 = nn.Linear(flattened_size, embedding_dim)  # mean
        self.fc2 = nn.Linear(flattened_size, embedding_dim)  # cov

        self.normal = torch.distributions.Normal(0, 1)

    def pass_through(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass that returns mean, cov
        """
        self.print_debug(f"\tInput shape: {x.shape}")
        x = self.prelu1(self.conv1(x))
        self.print_debug(f"\tShape after conv1: {x.shape}")

        # x = self.maxpool1(x)
        self.print_debug(f"\tShape after maxpool1: {x.shape}")

        x = self.prelu2(self.conv2(x))
        self.print_debug(f"\tShape after conv2: {x.shape}")

        # x = self.maxpool2(x)
        self.print_debug(f"\tShape after maxpool2: {x.shape}")

        x = self.prelu3(self.conv3(x))
        self.print_debug(f"\tShape after conv3: {x.shape}")

        # Flatten the tensor
        x = x.view(x.size(0), -1)  # batch_size x flattened_size
        self.print_debug(f"\tShape after flattening: {x.shape}")

        # Pass through the fully connected layer
        mean = self.fc1(x)
        cov = self.fc2(x)

        return mean, cov

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the CNN encoder

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, 1, x_dim, y_dim)

        Returns
        -------
        Tensor
            Mean of the latent distribution
        Tensor
            Covariance of the latent distribution
        """
        self.print_debug("\n\n###### ENCODER FORWARD PASS ######")

        mean, cov = self.pass_through(x)

        # Reparameterization trick
        eta = self.normal.sample(mean.shape)

        x = mean + torch.exp(cov / 2) * eta

        return x

    def predict(self, x: Tensor) -> Tensor:
        """
        Predicts the mean and covariance of the latent distribution

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, 1, x_dim, y_dim)

        Returns
        -------
        Tensor
            Mean of the latent distribution
        Tensor
            Covariance of the latent distribution
        """
        return self.pass_through(x)

    def print_debug(self, message):
        """
        Prints debug messages if debug mode is enabled
        """
        if self.debug:
            print(message)
