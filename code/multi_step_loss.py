import torch
import torch.nn as nn


class MultiStepLoss(nn.Module):
    """
    Loss function for the multi-step prediction
    """

    def __init__(
        self,
        state_loss_function=nn.MSELoss(),
        latent_loss_function=nn.MSELoss(),
        alpha=0.01,
        debug=False,
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
        super(MultiStepLoss, self).__init__()
        self.state_loss_function = state_loss_function
        self.latent_loss_function = latent_loss_function
        self.alpha = alpha
        self.debug = debug

    def forward(self, model, map, trajectory):
        """
        Forward pass of the loss function
        """
        self.print_debug("### MultiStepLoss Forward Pass ###")
        # Get rid of batch dimension
        trajectory = trajectory.squeeze(0)  # 1 x 3 x length
        map = map.squeeze(0)  # 1 x 2l x 2l

        self.print_debug(f"\tMap shape: {map.shape}")
        self.print_debug(f"\tTrajectory shape: {trajectory.shape}")

        goal_pose = trajectory[:, -1]
        sequence_length = trajectory.shape[1]

        map_views_tensor = self.extract_map_views(map, trajectory)  # LENGTH x 1 x l x l
        latent_maps = model.encode(map_views_tensor)  # (LENGTH x latent_dim)

        self.print_debug(f"\tmap_views_tensor shape: {map_views_tensor.shape}")
        self.print_debug(f"\tLatent maps shape: {latent_maps.shape}")

        # Batch predict the ground truth rollout
        start_pose_thetas = trajectory[2, :-1].unsqueeze(1)  # Length x 1
        goal_pose_diffs = goal_pose.unsqueeze(1) - trajectory[:, :-1]
        goal_pose_diffs[2, :] = goal_pose[2]
        goal_pose_diffs = goal_pose_diffs.T

        self.print_debug(f"\tstart_pose_thetas shape: {start_pose_thetas.shape}")
        self.print_debug(f"\tgoal_pose_diffs shape: {goal_pose_diffs.shape}")

        predictions = model.motion_planning_net(
            latent_maps[:-1], start_pose_thetas, goal_pose_diffs
        )

        self.print_debug(f"\tPredictions shape: {predictions.shape}")

        # Calculate the pose loss
        rollout_loss = self.state_loss_function(predictions, trajectory[:, 1:].T)

        reconstructed_maps = model.decode(latent_maps[:-1])
        reconstruction_loss = self.latent_loss_function(
            reconstructed_maps, map_views_tensor[:-1]
        )

        return rollout_loss + self.alpha * reconstruction_loss

    def extract_map_views(self, map, trajectory):
        """
        Extracts the map views around the trajectory points
        Collates into a tensor of shape (LENGTH, 1, l, l)
        """
        map_views = []
        for i in range(trajectory.shape[1]):
            int_pose = trajectory[:, i].int()
            map_view = map[
                :,
                int_pose[0] + 6 : int_pose[0] + 18,
                int_pose[1] + 6 : int_pose[1] + 18,
            ]
            map_views.append(map_view.unsqueeze(0))  # Add batch dimension

        return torch.cat(map_views, dim=0)  # convert to tensor along batch dim

    def print_debug(self, message):
        """
        Prints the debug message if debug is enabled
        """
        if self.debug:
            print(message)
