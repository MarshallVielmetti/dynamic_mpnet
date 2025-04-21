import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch


def PAD_MAPS(maps, MAP_PADDING):
    """
    Pads the maps to (60, 60) with 1s (occupied)
    """
    return torch.from_numpy(
        np.pad(
            maps,
            (
                (0, 0),
                (MAP_PADDING, MAP_PADDING),
                (MAP_PADDING, MAP_PADDING),
            ),
            mode="constant",
            constant_values=1,
        )
    )


class MultiStepTrajectoryDataset(Dataset):
    """
    Dataset for multi-step trajectory learning.

    Each sample consists of the following:
    - map: The occupancy grid map (2D tensor), sliced to 12x12 around the
           returned trajectory point.
    - trajectory: The trajectory points (3D tensor), where each point is
                  represented as (x, y, theta), transformed to be relative
                  to the center of the map
    Inputs to the model will be the map, first, and last points
    of the trajectory.
    """

    def __init__(
        self,
        maps: list[np.ndarray],
        trajectories: list[list[np.ndarray]],
        OUTPUT_MAP_SIZE=12,
        MULTI_STEP_SIZE=8,
        MULTI_STEP_SAMPLE_SKIP=4,
    ):
        """
        Initializes the dataset with maps and trajectories

        Parameters
        ----------
        maps : list[np.ndarray]
            The occupancy grid maps. The input shape is (num_maps, 48, 48)
            We pad to (60, 60) with 1s (occupied)
        trajectories : list[list[Tensor]]
            The trajectories. The first index corresponds to the map, the second index
            is ID of the trajectory. Each point on the trajectory is a 3x1 Tensor (x, y, theta)
        """
        print(f"Creating MultiStepTrajectoryDatatset")

        self.OUTPUT_MAP_SIZE = OUTPUT_MAP_SIZE
        self.MULTI_STEP_SIZE = MULTI_STEP_SIZE
        self.MAP_PADDING = OUTPUT_MAP_SIZE // 2

        # pad maps and convert to tensor
        self.maps = PAD_MAPS(maps, self.MAP_PADDING)

        self.data_points = []
        for map_id, map_trajectories in enumerate(trajectories):
            for trajectory in map_trajectories:
                # Skip trajectories that are too short
                if len(trajectory) < MULTI_STEP_SIZE:
                    continue

                # Sample multi-step trajectory data
                for sample_start in range(
                    0, len(trajectory) - MULTI_STEP_SIZE, MULTI_STEP_SAMPLE_SKIP
                ):
                    sampled_points = trajectory[
                        sample_start : sample_start + MULTI_STEP_SIZE
                    ]

                    if sampled_points.shape[1] != 3:
                        print(f"Sampled points shape mismatch: {sampled_points.shape}")
                        continue

                    # Transform the trajectory points to be relative to the center of the map
                    zero_point = sampled_points[0]
                    zero_point[0] = int(zero_point[0])
                    zero_point[1] = int(zero_point[1])

                    sampled_points = sampled_points - zero_point

                    map_start_idx = (
                        zero_point[0] - self.OUTPUT_MAP_SIZE // 2 + self.MAP_PADDING,
                        zero_point[1] - self.OUTPUT_MAP_SIZE // 2 + self.MAP_PADDING,
                    )
                    map_start_idx = (
                        int(map_start_idx[0]),
                        int(map_start_idx[1]),
                    )

                    for point in sampled_points:
                        if (
                            abs(point[0]) > self.MAP_PADDING
                            or abs(point[1]) > self.MAP_PADDING
                        ):
                            continue  # skip this sampling / don't add to dataset

                    # Convert sampled_points to a Tensor
                    sampled_points = torch.from_numpy(sampled_points).float().T

                    # Store the trajectory and its corresponding map
                    self.data_points.append((map_id, map_start_idx, sampled_points))

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        map_idx = self.data_points[idx][0]
        # print(f"Map ID: {map_idx}")
        map_start_idx = self.data_points[idx][1]
        # print(f"Map Start: {map_start_idx}")
        sampled_points = self.data_points[idx][2]
        # print(f"Got sample points")

        # Extract the map and trajectory points
        map_view = self.maps[map_idx][
            map_start_idx[0] : map_start_idx[0] + self.OUTPUT_MAP_SIZE,
            map_start_idx[1] : map_start_idx[1] + self.OUTPUT_MAP_SIZE,
        ]

        if (map_view.shape[0] != self.OUTPUT_MAP_SIZE) or (
            map_view.shape[1] != self.OUTPUT_MAP_SIZE
        ):
            print(f"Map view shape mismatch: {map_view.shape}")
            print(f"MAP_ID: {map_idx}")
            print(f"MAP_START: {map_start_idx}")
            raise ValueError(
                f"Map view shape mismatch: {map_view.shape} != {self.OUTPUT_MAP_SIZE}"
            )

        # add batch dimension
        map_view = map_view.unsqueeze(0).type(torch.float32)

        return map_view, sampled_points


class SingleStepTrajectoryDataset(Dataset):
    """
    Dataset for single-step trajectory learning

    Each sample consists of the following:
    - map: The occupancy grid map (2D tensor), sliced to 12x12 around the
           returned trajectory point.
    - Goal Pose: The goal pose (3D tensor), where each point is
                  represented as (x, y, theta), transformed to be relative
                  to the center of the map
    - Target Pose: The target pose, relative to the center of the map, to be predicted by the model (one timestep ahead)
    """

    def __init__(
        self,
        input_maps: list[np.ndarray],
        trajectories: list[list[np.ndarray]],
        OUTPUT_MAP_SIZE=12,
        SAMPLE_SKIPAHEAD=4,
    ):
        """
        Initializes the dataset with maps and trajectories

        Parameters
        ----------
        maps : list[np.ndarray]
            The occupancy grid maps. The input shape is (num_maps, 48, 48)
            We "pad" to (60, 60) with 1s (occupied)
        trajectories : list[list[Tensor]]
            The trajectories. The first index corresponds to the map, the second index
            is ID of the trajectory. Each point on the trajectory is a 3x1 Tensor (x, y, theta)
        OUTPUT_MAP_SIZE : int
            The size of the output map (default: 12)
        SAMPLE_SKIPAHEAD : int
            The number of steps to skip ahead when sampling the trajectory (default: 4)
        """

        self.OUTPUT_MAP_SIZE = OUTPUT_MAP_SIZE
        self.SAMPLE_SKIPAHEAD = SAMPLE_SKIPAHEAD
        self.MAP_PADDING = OUTPUT_MAP_SIZE // 2
        self.BASE_MAP_SIZE = 48
        self.MAP_SIZE = self.BASE_MAP_SIZE + 2 * self.MAP_PADDING

        # pad maps and convert to tensor
        self.maps = PAD_MAPS(input_maps, self.MAP_PADDING)

        # map_id, map_start_idx, start_theta, target_pose, goal_pose
        # target pose is the pose to predict
        # goal pose is the pose given as input target to NN
        self.data_points = []
        for map_id, map_trajectories in enumerate(trajectories):
            for trajectory in map_trajectories:
                # Skip trajectories that are too short
                if len(trajectory) < self.SAMPLE_SKIPAHEAD:
                    continue

                # Sample multi-step trajectory data
                for sample_start in range(
                    0, len(trajectory) - SAMPLE_SKIPAHEAD, SAMPLE_SKIPAHEAD
                ):

                    # Pose to calculate offset from
                    start_pose = trajectory[sample_start]
                    start_theta = start_pose[2]  # only part of the start pose we need

                    # Skip if the start pose is outside of the map (?)
                    if (
                        start_pose[0] < 0
                        or start_pose[0] > self.BASE_MAP_SIZE
                        or start_pose[1] < 0
                        or start_pose[1] > self.BASE_MAP_SIZE
                    ):
                        continue

                    target_pose = trajectory[sample_start + self.SAMPLE_SKIPAHEAD]
                    goal_pose = None

                    # Progress down trajectory to find the first point outside of the 12x12 padded map, then
                    # return the last point inside the map
                    for i in range(sample_start, len(trajectory)):
                        if (
                            abs(trajectory[i][0] - start_pose[0]) > self.MAP_PADDING
                            or abs(trajectory[i][1] - start_pose[1]) > self.MAP_PADDING
                        ):
                            goal_pose = trajectory[i - 1]
                            break

                    # If trajectory doesn't leave the map at this point, break
                    if goal_pose is None:
                        break

                    # Transform the trajectory points to be relative to the start pose of the map
                    target_pose[0:2] = target_pose[0:2] - start_pose[0:2]
                    goal_pose[0:2] = goal_pose[0:2] - start_pose[0:2]

                    # Find relative point to transform the map
                    zero_point = start_pose[0:2]

                    map_start_idx = (
                        zero_point[0] - self.OUTPUT_MAP_SIZE // 2 + self.MAP_PADDING,
                        zero_point[1] - self.OUTPUT_MAP_SIZE // 2 + self.MAP_PADDING,
                    )
                    map_start_idx = (
                        int(map_start_idx[0]),
                        int(map_start_idx[1]),
                    )

                    if (map_start_idx[0] < 0) or (map_start_idx[1] < 0):
                        print(f"Map start index out of bounds: {map_start_idx}")
                        print(f"MAP_ID: {map_id}")
                        print(f"START POITN: {start_pose}")
                        continue

                    # Convert anything that will be used as a model input to tensors
                    start_theta = torch.tensor(start_theta).float()
                    target_pose = torch.from_numpy(target_pose).float()
                    goal_pose = torch.from_numpy(goal_pose).float()

                    # Store the trajectory and its corresponding map
                    self.data_points.append(
                        (map_id, map_start_idx, start_theta, target_pose, goal_pose)
                    )

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        # Map information
        map_idx = self.data_points[idx][0]
        map_start_idx = self.data_points[idx][1]

        # Sample information
        start_theta = self.data_points[idx][2]
        target_pose = self.data_points[idx][3]
        goal_pose = self.data_points[idx][4]

        # Extract the map and trajectory points
        map_view = self.maps[map_idx][
            map_start_idx[0] : map_start_idx[0] + self.OUTPUT_MAP_SIZE,
            map_start_idx[1] : map_start_idx[1] + self.OUTPUT_MAP_SIZE,
        ]

        if (map_view.shape[0] != self.OUTPUT_MAP_SIZE) or (
            map_view.shape[1] != self.OUTPUT_MAP_SIZE
        ):
            print(f"Map view shape mismatch: {map_view.shape}")
            print(f"MAP_ID: {map_idx}")
            print(f"MAP_START: {map_start_idx}")
            raise ValueError(
                f"Map view shape mismatch: {map_view.shape} != {self.OUTPUT_MAP_SIZE}"
            )

        map_view = map_view.unsqueeze(0).type(torch.float32)

        return map_view, start_theta, target_pose, goal_pose
