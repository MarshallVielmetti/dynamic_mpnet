import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    Defines a CNN encoder for an OGM
    """

    def __init__(self, dimensions: tuple, embedding_dim, debug=False):
        """
        Initializes the CNN encoder

        Parameters
        ----------
        dimensions : tuple
            Dimensions of the input grid (x_dim, y_dim)
        """
        super(CNNEncoder, self).__init__()
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
        self.print_debug("\n\n###### ENCODER FORWARD PASS ######")
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
        x = self.fc1(x)
        self.print_debug(f"\tShape after fc1: {x.shape}")

        return x

    def print_debug(self, message):
        """
        Prints debug messages if debug mode is enabled
        """
        if self.debug:
            print(message)


class CNNDecoder(nn.Module):
    """
    Defines a CNN decoder for an OGM

    Takes a latent vector of shape (batch_size, embedding_dim) and outputs
    a tensor of shape (batch_size, 1, x_dim, y_dim)
    """

    def __init__(self, dimensions: tuple, embedding_dim, debug=False):
        """
        Initializes the CNN decoder

        Parameters
        ----------
        dimensions : tuple
            Dimensions of the input grid (x_dim, y_dim)
        """
        super(CNNDecoder, self).__init__()

        self.debug = debug

        self.dimensions = dimensions
        self.embedding_dim = embedding_dim

        self.fc1 = nn.Linear(embedding_dim, 3200)
        self.prelu1 = nn.PReLU()

        self.conv_transpose1 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=1, padding=1
        )
        self.prelu2 = nn.PReLU()
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_transpose2 = nn.ConvTranspose2d(
            16, 8, kernel_size=3, stride=1, padding=0
        )
        self.prelu3 = nn.PReLU()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_transpose3 = nn.ConvTranspose2d(
            8, 1, kernel_size=5, stride=1, padding=2
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the CNN decoder

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, embedding_dim)

        Returns
        -------
        Tensor
            Output tensor after passing through the CNN layers (batch_size, 1, x_dim, y_dim)
        """
        self.print_debug("\n\n###### DECODER FORWARD PASS ######")
        self.print_debug(f"\tInput shape: {x.shape}")

        x = self.prelu1(self.fc1(x))

        # Reshape the input tensor to match the expected input shape for the first layer
        x = x.view(x.size(0), 32, 10, 10)  # batch_size x 32 x 2 x 2
        self.print_debug(f"\tShape after reshaping: {x.shape}")

        x = self.prelu2(self.conv_transpose1(x))
        self.print_debug(f"\tShape after conv_transpose1: {x.shape}")

        # x = self.upsample1(x)
        # self.print_debug(f"\tShape after upsample1: {x.shape}")

        x = self.prelu3(self.conv_transpose2(x))
        self.print_debug(f"\tShape after conv_transpose2: {x.shape}")

        # x = self.upsample2(x)
        # self.print_debug(f"\tShape after upsample2: {x.shape}")

        x = self.conv_transpose3(x)
        self.print_debug(f"\tShape after conv_transpose3: {x.shape}")

        # Apply sigmoid activation to the output
        x = torch.sigmoid(x)
        self.print_debug(f"\tShape after sigmoid: {x.shape}")

        return x

    def print_debug(self, message):
        """
        Prints debug messages if debug mode is enabled
        """
        if self.debug:
            print(message)


import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MapDataset(Dataset):
    """
    Simple dataset class
    """

    def __init__(self, Xs):
        self.Xs = Xs

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        x_i = self.Xs[idx]
        return x_i, x_i


def get_data_loaders(folder="code/maps", train_split=0.8):
    """
    Loads the maps from the specified folder and returns a train and val data loader
    """
    print(f"Loading maps from {folder}...")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} is not a directory.")

    # Load the maps from the specified folder
    Xs = []
    files = os.listdir(folder)
    for file in files:
        if file.endswith(".npy"):
            # Load the numpy array and append it to the list
            x = np.load(os.path.join(folder, file))
            x = Tensor(x).unsqueeze(0)  # 1 x x_dim x y_dim
            Xs.append(x)

    if len(Xs) == 0:
        raise ValueError(f"No .npy files found in {folder}.")

    # Convert into a MapDataset
    map_dataset = MapDataset(Xs)

    # Split the dataset into training and validation sets
    train_size = int(len(Xs) * train_split)
    val_size = len(Xs) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        map_dataset, [train_size, val_size]
    )

    print(f"Train size: {train_size}, Validation size: {val_size}")
    print(f"Train dataset: {train_dataset}, Validation dataset: {val_dataset}")

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def train_autoencoder(
    encoder: CNNEncoder,
    decoder: CNNDecoder,
    train_loader,
    val_loader,
    lr=0.001,
    num_epochs=100,
):
    """
    Trains the autoencoder using the specified encoder and decoder

    Parameters
    ----------
    encoder : CNNEncoder
        The encoder model
    decoder : CNNDecoder
        The decoder model
    train_loader : DataLoader
        The DataLoader for the training set
    val_loader : DataLoader
        The DataLoader for the validation set
    lr : float
        The learning rate for the optimizer
    num_epochs : int
        The number of epochs to train for
    """

    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters()},
            {"params": decoder.parameters()},
        ],
        lr=lr,
    )

    train_losses = []
    val_losses = []

    for epoch_i in range(num_epochs):
        train_loss_i = train_step(encoder, decoder, train_loader, optimizer)
        val_loss_i = val_step(encoder, decoder, val_loader)

        if epoch_i % 50 == 0:
            print(
                f"Epoch {epoch_i + 1}/{num_epochs} | Train Loss: {train_loss_i:.4f} | Val Loss: {val_loss_i:.4f}"
            )

        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

    return train_losses, val_losses


def train_step(encoder, decoder, train_loader, optimizer) -> float:
    """
    Performs a single training step

    Parameters
    ----------
    encoder : CNNEncoder
        The encoder model
    decoder : CNNDecoder
        The decoder model
    train_loader : DataLoader
        The DataLoader for the training set
    optimizer : torch.optim.Optimizer
        The optimizer for the model
    """
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Model forward pass
        latent = encoder(data)
        y_hat = decoder(latent)

        optimizer.zero_grad()

        loss = F.mse_loss(target, y_hat) + F.l1_loss(target, y_hat) * 0.1

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def val_step(encoder, decoder, val_loader) -> float:
    """
    Performs a single validation step

    Parameters
    ----------
    encoder: CNNEncoder
        The encoder model
    decoder: CNNDecoder
        The decoder model
    val_loader: DataLoader
        The DataLoader for the validation set
    """
    val_loss = 0

    encoder.eval()
    decoder.eval()

    for batch_idx, (data, target) in enumerate(val_loader):
        latent = encoder(data)
        y_hat = decoder(latent)

        loss = F.mse_loss(target, y_hat) + F.l1_loss(target, y_hat) * 0.1

        val_loss += loss.item()

    return val_loss / len(val_loader)
