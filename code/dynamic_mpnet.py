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

    def __init__(self, map_dim: tuple, embedding_dim: int, output_steps: int, debug=False):
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


def train_dynamics_step(
    model: DynamicMPNet,
    encoder: CNNEncoder,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module = nn.MSELoss(),
) -> float:
    """
    Trains the dynamics model for one step

    Parameters
    ----------
    model : DynamicMPNet
        The dynamics model
    encoder : CNNEncoder
        The encoder model
    train_loader : DataLoader
        The training data loader
    optimizer : torch.optim.Optimizer
        The optimizer
    criterion : nn.Module
        The loss function
    Returns
    -------
    float
        The average loss for the step
    """
    train_loss = 0
    model.train()

    for batch_idx, (map, trajectory) in enumerate(train_loader):

        # Encode the map
        latent = encoder(map)  # B x 32

        start = trajectory[:, :, 0].reshape(-1, SHAPE_DIM)  # B x SHAPE
        goal = trajectory[:, :, -1].reshape(-1, SHAPE_DIM)  # B x SHAPE

        input = torch.cat(
            (latent, start, goal),
            dim=1,
        )  # B x (32 + 2 * SHAPE_DIM)

        output = model(input)  # B x (SHAPE_DIM * output_steps)

        # Transform the output to the correct shape - B x SHAPE_DIM x output_steps
        output = output.view(-1, SHAPE_DIM, model.output_steps)

        # Run the optimization step
        optimizer.zero_grad()
        loss = criterion(output, trajectory)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate_dynamics_step(
    model: DynamicMPNet,
    encoder: CNNEncoder,
    val_loader,
    criterion: nn.Module = nn.MSELoss(),
) -> float:
    """
    Performs a single validation step of the dynamics model

    Parameters
    ----------
    model : DynamicMPNet
        The dynamics model
    encoder : CNNEncoder
        The encoder model
    val_loader : DataLoader
        The validation data loader
    criterion : nn.Module
        The loss function
    """

    val_loss = 0
    encoder.eval()
    model.eval()

    with torch.no_grad():
        for batch_idx, (map, trajectory) in enumerate(val_loader):
            latent = encoder(map)

            start = trajectory[:, :, 0].reshape(-1, SHAPE_DIM)  # B x SHAPE
            goal = trajectory[:, :, -1].reshape(-1, SHAPE_DIM)  # B x SHAPE

            input = torch.cat(
                (latent, start, goal),
                dim=1,
            )

            output = model(input)
            output = output.view(-1, SHAPE_DIM, model.output_steps)

            loss = criterion(output, trajectory)

            val_loss += loss.item()

    return val_loss / len(val_loader)


def train_model_multi_step(
    model: DynamicMPNet,
    encoder: CNNEncoder,
    train_loader,
    val_loader,
    lr: float = 0.001,
    num_epochs: int = 100,
) -> tuple[list, list]:
    """
    Trains the dynamics model using the given parameters and premade encoder
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Loss Function
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch_i in range(num_epochs):
        train_loss_i = train_dynamics_step(
            model,
            encoder,
            train_loader,
            optimizer,
            criterion=criterion,
        )
        val_loss_i = validate_dynamics_step(
            model, encoder, val_loader, criterion=criterion
        )

        if epoch_i % 10 == 0:
            print(
                f"Epoch {epoch_i} | Train Loss: {train_loss_i:.4f} | Val Loss: {val_loss_i:.4f}"
            )

        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

    return train_losses, val_losses


def train_model_single_step(
    model: model
)