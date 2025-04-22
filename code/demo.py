import torch
import numpy as np
import matplotlib.pyplot as plt

from dynamic_mpnet import DynamicMPNet
from grid_environment import GridEnvironment
from occupancy_grid_generator import OccupancyGridGenerator
from dubins import Dubins

MAP_DIM = (12, 12)
EMBEDDING_DIM = 32
NUM_OBSTACLES = 4


def main():
    print("Expected Time to Run: 30 seconds")

    # Load the model
    model_path = "code/models/dynamic_mpnet.pth"
    model = DynamicMPNet(MAP_DIM, EMBEDDING_DIM)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ## Generate a random map using an input seed ##
    random_seed = 123123
    np.random.seed(random_seed)
    ogm_generator = OccupancyGridGenerator(MAP_DIM, NUM_OBSTACLES)
    map = ogm_generator.sample()
    map_tensor = torch.as_tensor(map, dtype=torch.float32).unsqueeze(0)

    # generate map
    env = GridEnvironment(map)

    # sample start and goal positions from free space
    start = torch.as_tensor(
        env.random_free_space(include_theta=True), dtype=torch.float32
    )
    goal = torch.as_tensor(
        env.random_free_space(include_theta=True), dtype=torch.float32
    )

    # Plot the env, start, and goal
    env.plot(close=False, display=False)
    plt.scatter(start[0], start[1], c="green", marker="o", label="Start")
    plt.scatter(goal[0], goal[1], c="red", marker="x", label="Goal")
    plt.legend()
    plt.show()
    return

    goal_reached = False
    trajectory = []
    curr_point = start
    while not goal_reached:
        # predict next point using most recent point
        next = model(map_tensor, curr_point[2], goal)
        next = next.squeeze(0).cpu().detach().numpy()
        print(f"Next point: {next}")

        # steer to point (fit dubins and verify collision free)
        # add the point to the trajectory

        # try to connect steer point to goal
        # if can, addgoal to trajectory and return

    # Plot the environment

    # plot the start and end points as green and red arrows

    # plot a simple dubins path
    # plot the trajectory, interpolating dubins path between points
    # each interior point of the trajectory should be an arrow

    # could also run RRT* to find a path? And record the timing information? If I end up having time or caring enough


if __name__ == "__main__":
    main()
