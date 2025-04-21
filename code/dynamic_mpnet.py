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

    def __init__(
        self, map_dim: tuple, embedding_dim: int, output_steps: int, debug=False
    ):
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
        self.output_steps = output_steps
        self.debug = debug

        self.encoder = CNNEncoder(map_dim, embedding_dim, debug=debug)
        self.decoder = CNNDecoder(map_dim, embedding_dim, debug=debug)

        # Define the model architecture
        self.fc1 = nn.Linear(embedding_dim + 2 * SHAPE_DIM, 128)
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

        self.fc6 = nn.Linear(128, SHAPE_DIM * output_steps)
        self.tanh = nn.Tanh()

    def forward(self, map, start_theta, goal) -> Tensor:
        """
        Forward pass of the dynamics network

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, embedding_dim + 2*3)

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, 3 * output_steps)
        """

        self.print_debug("\n\n ##### Dynamics Forwards Pass ######")
        self.print_debug(f"Input shape: {x.shape}")

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
        self.print_debug(f"Motion Planning Net: Latent map shape: {latent_map.shape}")
        self.print_debug(f"Motion Planning Net: Goal shape: {goal.shape}")

        # Combine the inputs into a single tensor
        x = torch.cat((latent_map, start_theta, goal), dim=1)
        self.print_debug(f"Motion Planning Net: Combined input shape: {x.shape}")

        x = self.dropout1(self.prelu1(self.fc1(x)))
        x = self.dropout2(self.prelu2(self.fc2(x)))
        x = self.dropout3(self.prelu3(self.fc3(x)))
        x = self.dropout4(self.prelu4(self.fc4(x)))

        x = self.prelu5(self.fc5(x))
        x = self.tanh(self.fc6(x))

        self.print_debug(f"Motion Planning Net: Output shape: {x.shape}")
        return x

    def print_debug(self, message):
        if self.debug:
            print(message)


class SingleStepLoss(nn.Module):
    """
    Loss function for the single step prediction
    """

    def __init__(
        self, state_loss_function=nn.MSELoss, latent_loss_function=nn.MSELoss, alpha=0.1
    ):
        """
        Initializes the loss function with the given parameters

        Parameters
        ----------
        state_loss_function : nn.Module
            The loss function for the state prediction
        latent_loss_function : nn.Module
            The loss function for the latent prediction
        alpha : float
            The weight for the latent loss
        """
        super(SingleStepLoss, self).__init__()
        self.state_loss_function = state_loss_function
        self.latent_loss_function = latent_loss_function
        self.alpha = alpha

    def forward(self, model, map, start_theta, goal_pose, target_pose):
        """
        Forward pass of the loss function

        Uses the two terms
        - State Loss: MSE loss between predicted next state and ground truth
        - Reconstruction Loss: MSE between original and reconstructed map
        """
        # Get the latent representation of the map
        latent_map = model.encode(map)

        # Get the predicted trajectory from the model
        predicted_step = model.motion_planning_net(latent_map, start_theta, goal_pose)

        # Calculate the state loss and latent loss
        state_loss = self.state_loss_function(predicted_step, target_pose)
        latent_loss = self.latent_loss_function(map, latent_map)

        # Combine the losses using the alpha parameter
        return state_loss + self.alpha * latent_loss


def train_dynamics_step(
    model: DynamicMPNet,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module = SingleStepLoss(),
) -> float:
    """
    Trains the dynamics model for one step
    """
    train_loss = 0
    model.train()

    for batch_idx, (map, start_theta, target_pose, goal_pose) in enumerate(
        train_loader
    ):
        # Reset the optimizer
        optimizer.zero_grad()

        # Compute the loss
        loss = criterion(model, map, start_theta, goal_pose, target_pose)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate_dynamics_step(
    model: DynamicMPNet,
    val_loader,
    criterion: nn.Module = SingleStepLoss(),
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


def train_model_single_step(
    model: DynamicMPNet,
    train_loader,
    val_loader,
    lr: float = 0.001,
    num_epochs: int = 100,
) -> tuple[list, list]:
    """
    Trains the dynamics model using the given parameters on single step data
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    criterion = SingleStepLoss()

    train_losses = []
    val_losses = []

    for epoch_i in range(num_epochs):
        train_loss_i = train_dynamics_step(
            model,
            train_loader,
            optimizer,
            criterion=criterion,
        )

        val_loss_i = validate_dynamics_step(model, val_loader, criterion=criterion)

        if epoch_i % 10 == 0:
            print(
                f"Epoch {epoch_i} | Train Loss: {train_loss_i:.4f} | Val Loss: {val_loss_i:.4f}"
            )

        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

    return train_losses, val_losses
