import numpy as np
import torch

from dynamic_mpnet import DynamicMPNet
from grid_environment import GridEnvironment
from dubins import Dubins


class DynamicMPNetPlanner:
    def __init__(
        self, model: DynamicMPNet, num_samples=10, sample_spacing=0.1, dubins_radius=2
    ):
        self.model = model  # the MPNet model
        self.num_samples = num_samples  # number of samples to take to attempt to complete the trajectory
        self.sample_spacing = sample_spacing  # spacing between sampled points

        self.dubins_planner = Dubins(
            radius=dubins_radius, point_separation=self.sample_spacing
        )

        self.debug = False

    def plan(self, env: GridEnvironment, x_from: np.ndarray, x_goal: np.ndarray):
        """
        Plan a path from x0 to xg using the dynamic MPNet model. Implements Algorithm 1 from the paper

        Parameters
        ----------
        map : np.ndarray
            The occupancy grid map.
        x0 : np.ndarray
            The start state.
        xg : np.ndarray
            The goal state.

        Returns
        -------
        trajectory : np.ndarray
            The planned trajectory.
        """
        tau = [x_from[0:2]]
        x_curr = x_from

        # Transform the map into an egocentric frame, where the "from" state is the center of the map
        # the map consists of a 12x12 grid, padded to 24x24 with obstacles
        map = self.prep_grid(x_from, env.grid)

        for i in range(self.num_samples):
            c_hat = self.transform(x_from, x_curr, map)
            x_temp = self.net(x_curr, x_goal, c_hat)

            self.print_debug(f"sampled point: {x_temp} given {x_curr} and {x_goal}")

            tau_temp = self.steer(x_curr, x_temp, env)

            # If failed to connect the points with collision free trajectory, try again
            if tau_temp is None:
                continue

            # if the trajectory is collision free, append it to the existing path
            # tau.append(x_temp[1:])
            tau.extend(tau_temp[1:])

            # Try to connect the last point of the trajectory to the goal
            tau_goal = self.steer(x_temp, x_goal, env)

            # If able to do so, return!
            if tau_goal is not None:
                # tau.append(tau_goal[1:])
                tau.extend(tau_goal[1:])
                return tau

            x_curr = x_temp

        return None

    def prep_grid(self, x_from: np.ndarray, map: np.ndarray):
        """
        Preprocess the occupancy grid map to be used as input to the model.

        This produces the blank 12x12 map, centered around the state, that will be used from now on
        as the base map

        Parameters
        ----------
        map : np.ndarray
            The occupancy grid map.

        Returns
        -------
        map : np.ndarray
            The preprocessed occupancy grid map.
        """

        # get an int version of x_from
        x_from = np.astype(x_from, np.int32)

        # first, pad the map wsth 1s (obstacles) to make it 24x24
        # this will allow us to sample a 12x12 grid from the map regardless of the from state
        padded_map = np.pad(
            map,
            ((6, 6), (6, 6)),
            mode="constant",
            constant_values=1,
        )

        # get the 12x12 map centered around the state
        # the offsets are what they are because we just padded the map by 6
        # so (0, 0) => (6, 6) in the padded map
        padded_map = padded_map[
            x_from[0] : x_from[0] + 12,
            x_from[1] : x_from[1] + 12,
        ]

        # now pad the map again to make it 24x24
        padded_map = np.pad(
            padded_map,
            ((6, 6), (6, 6)),
            mode="constant",
            constant_values=1,
        )

        return padded_map

    def transform(self, x_from: np.ndarray, x_curr: np.ndarray, map: np.ndarray):
        """
        Transform the map to the model input space.

        Parameters
        ----------
        x_from : np.ndarray
            The state around which the map is centered
        x_curr : np.ndarray
            The state to recenter and trim the map around
        map : np.ndarray
            The occupancy grid map. (24x24)

        Returns
        -------
        c_hat : np.ndarray
            The transformed map.
        """
        x_offset = x_curr - x_from
        x_offset = np.clip(x_offset, -6, 6)
        x_offset = np.astype(x_offset, np.int32)

        x_offset += np.array([12, 12, 0])

        # get 12x12 map centered around the state
        map = map[
            x_offset[0] - 6 : x_offset[0] + 6,
            x_offset[1] - 6 : x_offset[1] + 6,
        ]

        return map

    def net(self, x_curr: np.ndarray, x_goal: np.ndarray, c_hat: np.ndarray):
        """
        Call the MPNet model to get the next state.

        Parameters
        ----------
        x : np.ndarray
            The state.
        x_goal : np.ndarray
            The goal state.
        c_hat : np.ndarray
            The transformed state and map.

        Returns
        -------
        x_temp : np.ndarray
            The next state.
        """

        # Transform the goal state to be relative to the current state
        transformed_goal = x_goal - x_curr
        transformed_goal[2] = x_goal[2]  # but don't transform the theta value

        # Convert values to tensors
        theta_tensor = (
            torch.as_tensor(x_curr[2], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

        x_goal_tensor = torch.as_tensor(
            transformed_goal, dtype=torch.float32
        ).unsqueeze(0)

        map_tensor = (
            torch.as_tensor(c_hat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

        # model output is a tensor of shape (1, 3), and reflects the position offset
        model_output = self.model.forward(map_tensor, theta_tensor, x_goal_tensor)
        model_output = model_output.squeeze(0).cpu().detach().numpy()

        # offset the model output
        next_state = x_curr + model_output
        next_state[2] = model_output[2]  # but don't offset the theta value

        return next_state

    def steer(self, x_from: np.ndarray, x_to: np.ndarray, env: GridEnvironment):
        """
        Steer the state towards the next state. Attempts to fit a
        dubins path between the two points, samples along the path, and checks for collision

        Parameters
        ----------
        x_from : np.ndarray
            The starting state
        x_temp : np.ndarray
            The state to steer towards

        Returns
        -------
        tau_temp : np.ndarray
            The sampled trajectory, or None if no trajectory exists (e.g. not collision free)
        """
        # fit a dubins path between the two points
        tau_temp, _ = self.dubins_planner.dubins_path(x_from, x_to)

        # check for collision
        for point in tau_temp:
            if not env.is_free(point[0], point[1]):
                self.print_debug(f"Collision at point {point}")
                return None

        return tau_temp

    def add_to_path(self, tau: list, tau_temp: np.ndarray):
        """
        Add the sampled trajectory to the path. Will need to transform points back to original space

        Parameters
        ----------
        tau : list
            The path.
        tau_temp : np.ndarray
            The sampled trajectory.
        """
        pass

    def print_debug(self, message: str):
        if self.debug:
            print(message)
