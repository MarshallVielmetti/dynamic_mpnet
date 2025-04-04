import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    Defines a CNN encoder for an OGM
    """

    def __init__(self, dimensions: tuple, embedding_dim):
        """
        Initializes the CNN encoder

        Parameters
        ----------
        dimensions : tuple
            Dimensions of the input grid (x_dim, y_dim)
        """
        super(CNNEncoder, self).__init__()
        self.dimensions = dimensions

        # Define the CNN layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        # self.prelu3 = nn.PReLU()

        flattened_size = 

        self.fc1 = nn.Linear(flattened_size, embedding_dim)


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
            Output tensor after passing through the CNN layers
        """
        x = F.prelu(self.conv1(x))
        x = self.maxpool1(x)

        x = F.prelu(self.conv2(x))
        x = self.maxpool2(x)

        x = F.prelu(self.conv3(x))

        print(f"Shape after conv3: {x.shape}")
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        print(f"Shape after flattening: {x.shape}")

        # Pass through the fully connected layer
        x = self.fc1(x)
        print(f"Shape after fc1: {x.shape}")
        

        return x

class CNNDecoder(nn.Module):
    """
    Defines a CNN decoder for an OGM
    """

    def __init__(self, dimensions: tuple):
        """
        Initializes the CNN decoder

        Parameters
        ----------
        dimensions : tuple
            Dimensions of the input grid (x_dim, y_dim)
        """
        super(CNNDecoder, self).__init__()
        self.dimensions = dimensions

        # Define the CNN layers
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)
        self.prelu1 = nn.PReLU()

