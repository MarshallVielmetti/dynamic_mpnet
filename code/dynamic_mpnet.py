import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from map_encoder import CNNEncoder, CNNDecoder

SHAPE_DIM = 3


class DynamicMPNet(nn.Module):
    """
    Full NN model for dynamic trajectory prediction
    """

    def __init__(self, map_dim: tuple, embedding_dim: int, debug=False):
        """
        Initializes the model with the given parameters

        Six layer fully connected NN with PreLU and Dropout

        Parameters
        ----------
        map_dim : int
            The dimension of the map input
        embedding_dim : int
            The dimension of the embedded map
        output_steps : int
            The number of steps to predict
        debug : bool
            Whether to print debug information
        """
        super(DynamicMPNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.debug = debug

        self.encoder = CNNEncoder(map_dim, embedding_dim, debug=debug)
        self.decoder = CNNDecoder(map_dim, embedding_dim, debug=debug)

        # Define the model architecture
        self.fc1 = nn.Linear(embedding_dim + SHAPE_DIM + 1, 128)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(0.05)

        self.fc2 = nn.Linear(128, 256)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(0.05)

        self.fc3 = nn.Linear(256, 256)
        self.prelu3 = nn.PReLU()
        self.dropout3 = nn.Dropout(0.05)

        self.fc4 = nn.Linear(256, 256)
        self.prelu4 = nn.PReLU()
        self.dropout4 = nn.Dropout(0.05)

        self.fc5 = nn.Linear(256, 128)
        self.prelu5 = nn.PReLU()

        self.fc6 = nn.Linear(128, SHAPE_DIM)
        # self.tanh = nn.Tanh()

    def forward(self, map, start_theta, goal) -> Tensor:
        """
        Forward pass of the dynamics network

        Parameters
        ----------
        map: Tensor
            The map input of shape (batch_size, 1, map_dim, map_dim)
        start_theta: Tensor
            The start theta input of shape (batch_size, 1)
        goal: Tensor
            The goal input of shape (batch_size, 1)

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, 3 * output_steps)
        """
        self.print_debug("\n\n ##### Dynamics Forwards Pass ######")

        # Encode the map
        latent_map = self.encode(map)

        # Forward pass through the motion planning network
        return self.motion_planning_net(latent_map, start_theta, goal)

    def encode(self, map):
        """
        Encodes the map using the CNNEncoder
        """
        self.print_debug(f"Encode: Map shape: {map.shape}")
        # Reshape the map to the correct shape
        return self.encoder(map)

    def decode(self, latent):
        """
        Decodes the latent representation using the CNNDecoder
        """
        self.print_debug(f"Decode: Latent shape: {latent.shape}")
        # Reshape the latent to the correct shape
        reconstructed_map = self.decoder(latent)

        self.print_debug(f"Decode: Reconstructed map shape: {reconstructed_map.shape}")
        return reconstructed_map

    def motion_planning_net(self, latent_map, start_theta, goal):
        """
        Runs the motion planning network
        """
        # start_theta = start_theta.unsqueeze(1)
        self.print_debug("###### Motion Planning Net ######")
        self.print_debug(f"\tMotion Planning Net: Latent map shape: {latent_map.shape}")
        self.print_debug(
            f"\tMotion Planning Net: Start theta shape: {start_theta.shape}"
        )
        self.print_debug(f"\tMotion Planning Net: Goal shape: {goal.shape}")

        # Combine the inputs into a single tensor
        x = torch.cat((latent_map, start_theta, goal), dim=1)
        self.print_debug(f"Motion Planning Net: Combined input shape: {x.shape}")

        x = self.dropout1(self.prelu1(self.fc1(x)))
        x = self.dropout2(self.prelu2(self.fc2(x)))
        x = self.dropout3(self.prelu3(self.fc3(x)))
        x = self.dropout4(self.prelu4(self.fc4(x)))

        x = self.prelu5(self.fc5(x))
        x = self.fc6(x)

        self.print_debug(f"Motion Planning Net: Output shape: {x.shape}")
        return x

    def print_debug(self, message):
        if self.debug:
            print(message)


def train_dynamics_step(
    model: DynamicMPNet,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    Trains the dynamics model for one step
    """
    train_loss = 0
    model.train()

    print("Train dynamics step")

    for batch_idx, (train_data) in enumerate(train_loader):
        # Reset the optimizer
        optimizer.zero_grad()

        # Compute the loss
        loss = criterion(model, *train_data)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate_dynamics_step(
    model: DynamicMPNet, val_loader, criterion: nn.Module
) -> float:
    """
    Performs a single validation step of the dynamics model
    """
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (map, start_theta, target_pose, goal_pose) in enumerate(
            val_loader
        ):
            # Compute the loss
            loss = criterion(model, map, start_theta, target_pose, goal_pose)

            # increment
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train_model(
    model: DynamicMPNet,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 100,
) -> tuple[list, list]:
    """
    Trains the dynamics model using the given parameters on single step data
    """
    train_losses = []
    val_losses = []

    for epoch_i in range(num_epochs):
        train_loss_i = train_dynamics_step(model, train_loader, optimizer, criterion)

        # val_loss_i = validate_dynamics_step(model, val_loader, criterion)
        val_loss_i = 0
        print(
            f"EPOCH {epoch_i} | Train Loss: {train_loss_i:.4f} | Val Loss: {val_loss_i:.4f}"
        )

        if epoch_i % 10 == 0:
            print(
                f"Epoch {epoch_i} | Train Loss: {train_loss_i:.4f} | Val Loss: {val_loss_i:.4f}"
            )

        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

    return train_losses, val_losses
