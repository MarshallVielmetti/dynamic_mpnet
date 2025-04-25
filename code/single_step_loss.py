import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from map_encoder import CNNEncoder, CNNDecoder


class SingleStepLoss(nn.Module):
    """
    Loss function for the single step prediction
    """

    def __init__(
        self,
        state_loss_function=nn.MSELoss(),
        latent_loss_function=nn.MSELoss(),
        alpha=1.0,
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

    def forward(self, model, map, start_theta, target_pose, goal_pose):
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

        # Get the reconstructed map from the latent representation
        reconstructed_map = model.decode(latent_map)

        # Calculate the state loss and latent loss
        # print(f"Predicted: {predicted_step} vs. Target: {target_pose} ")
        state_loss = self.state_loss_function(predicted_step, target_pose)
        latent_loss = self.latent_loss_function(reconstructed_map, map)

        # print(f"State Loss: {state_loss.item()} | Latent Loss: {latent_loss.item()}")

        # Combine the losses using the alpha parameter
        return state_loss + self.alpha * latent_loss
