import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch.nn.functional as F
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

        Dataset structure:
        - map_id: The ID of the map
        - map_start_idx: the bottom left corner of the map to return
        - start_theta: The starting theta of the trajectory
        - sampled_points: The trajectory points (3D tensor)
        - goal_pose: The goal pose (3D tensor)
        """
        print(f"Creating MultiStepTrajectoryDatatset")

        self.BASE_MAP_SIZE = 48
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

                    # Progress down the trajectory to find the first point outside of
                    # the 12x12 padded map, then return the last point inside the map
                    offset = None
                    for i in range(sample_start, len(trajectory)):
                        if (
                            abs(trajectory[i][0] - start_pose[0]) > self.MAP_PADDING
                            or abs(trajectory[i][1] - start_pose[1]) > self.MAP_PADDING
                        ):
                            offset = i - 1
                            break

                    # If trajectory doesn't leave the map at this point, break
                    if offset is None:
                        break

                    # Sample the trajectory points
                    sample_traj = trajectory[sample_start:offset:MULTI_STEP_SAMPLE_SKIP]

                    non_int_zero_point = np.array(start_pose, dtype=np.float32)
                    non_int_zero_point[2] = 0

                    # Transform the trajectory points to be relative to the center of the map
                    zero_point = start_pose[0:2]
                    zero_point[0] = int(zero_point[0])
                    zero_point[1] = int(zero_point[1])

                    sample_traj = sample_traj - non_int_zero_point

                    map_start_idx = (
                        zero_point[0] - self.OUTPUT_MAP_SIZE // 2 + self.MAP_PADDING,
                        zero_point[1] - self.OUTPUT_MAP_SIZE // 2 + self.MAP_PADDING,
                    )
                    map_start_idx = (
                        int(map_start_idx[0]),
                        int(map_start_idx[1]),
                    )

                    for point in sample_traj:
                        if (
                            abs(point[0]) > self.MAP_PADDING
                            or abs(point[1]) > self.MAP_PADDING
                        ):
                            continue  # skip this sample / don't add to dataset

                    # Convert sampled_points to a Tensor
                    start_theta = torch.tensor(start_theta).float()
                    sample_traj = torch.from_numpy(sample_traj).float().T

                    # Store the trajectory and its corresponding map
                    self.data_points.append((map_id, map_start_idx, sample_traj))

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        map_idx = self.data_points[idx][0]
        map_start_idx = self.data_points[idx][1]
        sample_traj = self.data_points[idx][2]

        # Extract the map and trajectory points
        map_view = self.maps[map_idx][
            map_start_idx[0] : map_start_idx[0] + self.OUTPUT_MAP_SIZE,
            map_start_idx[1] : map_start_idx[1] + self.OUTPUT_MAP_SIZE,
        ]

        # Pad the map to (24, 24) with 1s (occupied)
        map_view = F.pad(
            map_view,
            (self.MAP_PADDING, self.MAP_PADDING, self.MAP_PADDING, self.MAP_PADDING),
            "constant",
            1,
        )

        # add batch dimension
        map_view = map_view.unsqueeze(0).type(torch.float32)

        return (map_view, sample_traj)


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

                    # Target pose is the pose to PREDICT
                    target_pose = trajectory[sample_start + self.SAMPLE_SKIPAHEAD]
                    goal_pose = None

                    # Progress down trajectory to find the first point outside of the 12x12 padded map, then
                    # return the last point inside the map
                    for i in range(sample_start + SAMPLE_SKIPAHEAD, len(trajectory)):
                        if (
                            abs(trajectory[i][0] - start_pose[0]) > self.MAP_PADDING
                            or abs(trajectory[i][1] - start_pose[1]) > self.MAP_PADDING
                        ):
                            goal_pose = trajectory[i - 1]
                            break

                    # If trajectory doesn't leave the map at this point, break
                    if goal_pose is None:
                        break

                    # Create a version of start pose with a zero theta

                    # Transform the trajectory points to be relative to the start pose of the map
                    target_pose[0:2] = target_pose[0:2] - start_pose[0:2]
                    goal_pose[0:2] = goal_pose[0:2] - start_pose[0:2]

                    # Make sure target and goal pose are within the map
                    if (
                        abs(target_pose[0]) > self.MAP_PADDING
                        or abs(target_pose[1]) > self.MAP_PADDING
                        or abs(goal_pose[0]) > self.MAP_PADDING
                        or abs(goal_pose[1]) > self.MAP_PADDING
                    ):
                        continue

                    # Find relative point to transform the map
                    zero_point = np.array(start_pose)
                    zero_point[2] = 0

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

        # TARGET IS THE POSE TO PREDICT
        # GOAL IS THE POSE TO USE AS INPUT
        return map_view, start_theta, target_pose, goal_pose
