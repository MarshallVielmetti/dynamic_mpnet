import torch
import torch.nn as nn

from dynamic_mpnet import DynamicMPNet


class MultiStepBatchLoss(nn.Module):
    """
    Batched loss function for the multi-step prediction
    """

    def __init__(
        self,
        state_loss_function=nn.MSELoss(),
        latent_loss_function=nn.MSELoss(),
        alpha=0.1,
    ):
        """
        Initializes the loss function with the given parameters

        Parameters
        ----------
        state_loss_function : nn.Module
            The loss function for the state prediction
        latent_loss_function : nn.Module
            The loss function for the latent representation
        alpha : float
            The weight for the latent loss
        """
        super(MultiStepBatchLoss, self).__init__()
        self.state_loss_function = state_loss_function
        self.latent_loss_function = latent_loss_function
        self.alpha = alpha

    def forward(self, model: DynamicMPNet, map, trajectory):
        """
        Forward pass of the loss function

        Parameters
        ----------
        model: DynamicMPNet
            model
        map: torch.Tensor
            (B, 1, 2*l, 2*l)
        trajectory: torch.Tensor
            (B, 3, )
        """
