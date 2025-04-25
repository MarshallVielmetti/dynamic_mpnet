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
        super(MultiStepLoss, self).__init__()
        self.state_loss_function = state_loss_function
        self.latent_loss_function = latent_loss_function
        self.alpha = alpha

    def forward(self, model, map, trajectory):
        """
        Forward pass of the loss function

        Uses the three loss terms
        - State Loss: MSE loss between predicted next state and ground truth
        - Reconstruction Loss: MSE between original and reconstructed map
        - Rollout Loss: MSE between predicted trajectory and ground truth

        First rolls out the trajectory using only the model
        Then does it again, using the model and the ground truth at each step

        Parameters
        ----------
        model is the model
        map is the map B, 1, 2*l, 2*l
        trajectory is the trajectory B, 3
        """
        # Get rid of batch dimension
        trajectory = trajectory.squeeze(0)  # 1 x 3 x length
        map = map.squeeze(0)  # 1 x 2l x 2l

        ## DOES NOT SUPPORT BATCH DIMENSIONS BECAUSE
        ## TRAJECTORIES CAN BE OF DIFFERENT LENGTHS

        # Get the goal pose that will be fed as input to the model
        goal_pose = trajectory[:, -1]

        rollout_poses = []
        curr_pose = trajectory[:, 0]
        for i in range(1, trajectory.shape[1]):
            # Get a view of the map around the curr_pose
            int_pose = curr_pose.int()

            map_view = map[
                :,
                int_pose[0] + 6 : int_pose[0] + 18,
                int_pose[1] + 6 : int_pose[1] + 18,
            ]

            curr_pose_zero_theta = curr_pose
            curr_pose_zero_theta[2] = 0

            predicted_step = model(
                map_view.unsqueeze(0),
                curr_pose[2].unsqueeze(0).unsqueeze(0),
                (goal_pose - curr_pose_zero_theta).unsqueeze(0),
            ).squeeze(0)

            # Append the predicted step to the rollout poses
            rollout_poses.append(predicted_step + curr_pose_zero_theta)
            curr_pose = predicted_step

        # Now do the same thing, but use the ground truth at each step
        # and also store the latent representation of the map
        latent_vectors = []
        rollout_poses_gt = []
        curr_pose = trajectory[:, 0]
        for i in range(1, trajectory.shape[1]):
            # Get a view of the map around the curr_pose
            int_pose = curr_pose.int()
            map_view = map[
                :,
                int_pose[0] + 6 : int_pose[0] + 18,
                int_pose[1] + 6 : int_pose[1] + 18,
            ]

            # Get the latent representation of the map
            latent_map = model.encode(map_view.unsqueeze(0)).squeeze(0)
            latent_vectors.append(latent_map)

            curr_pose_zero_theta = curr_pose
            curr_pose_zero_theta[2] = 0

            # Get the predicted step using the ground truth
            predicted_step = model.motion_planning_net(
                latent_map.unsqueeze(0),
                curr_pose[2].unsqueeze(0).unsqueeze(0),
                (goal_pose - curr_pose_zero_theta).unsqueeze(0),
            ).squeeze(0)

            # Append the predicted step to the rollout poses
            rollout_poses_gt.append(predicted_step + curr_pose_zero_theta)
            curr_pose = trajectory[:, i]

        # Now, calculate the losses
        rollout_poses = torch.stack(rollout_poses, dim=1)
        rollout_poses_gt = torch.stack(rollout_poses_gt, dim=1)

        rollout_loss = self.state_loss_function(rollout_poses, trajectory[:, 1:])
        rollout_loss += self.state_loss_function(rollout_poses_gt, trajectory[:, 1:])

        reconstruction_loss = 0
        for i in range(len(latent_vectors)):
            # Get the reconstructed map from the latent representation
            reconstructed_map = model.decode(latent_vectors[i].unsqueeze(0)).squeeze(0)

            int_pose = trajectory[:, i].int()

            ground_truth_map = map[
                :,
                int_pose[0] + 6 : int_pose[0] + 18,
                int_pose[1] + 6 : int_pose[1] + 18,
            ]

            # Calculate the reconstruction loss
            reconstruction_loss += self.latent_loss_function(
                reconstructed_map, ground_truth_map
            )

        return rollout_loss + self.alpha * reconstruction_loss
