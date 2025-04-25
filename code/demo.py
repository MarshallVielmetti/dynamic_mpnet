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


DUBINS = Dubins(radius=2, point_separation=0.1)


def check_path_collision(env, trajectory, gaol):
    """
    Check if the trajectory collides with any obstacles in the environment.
    """
    for i in range(len(trajectory) - 1):
        x_curr = trajectory[i]
        x_next = trajectory[i + 1]

        # Try to fit a dubins path between the two points
        path, _ = DUBINS.dubins_path(x_curr, x_next)

        # Make sure every point in path is collision free
        for point in path:
            if not env.is_free(point[0], point[1]):
                return False

    return True


def main():
    print("Expected Time to Run: 30 seconds")

    # Load the model
    model_path = "code/models/dynamic_mpnet.pth"
    model = DynamicMPNet(MAP_DIM, EMBEDDING_DIM)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ## Generate a random map using an input seed ##
    random_seed = 1
    np.random.seed(random_seed)
    ogm_generator = OccupancyGridGenerator(MAP_DIM, NUM_OBSTACLES)
    map = ogm_generator.sample()
    map = np.pad(
        map,
        ((6, 6), (6, 6)),
        mode="constant",
        constant_values=1,
    )

    map_tensor = torch.as_tensor(map, dtype=torch.float32).unsqueeze(0)

    # generate map
    env = GridEnvironment(map)

    # sample start and goal positions from free space
    # start = torch.as_tensor(
    #     env.random_free_space(include_theta=True), dtype=torch.float32
    # )
    # start = np.array([12, 12, 0])
    start = torch.tensor([12, 12, np.pi / 2], dtype=torch.float32)
    goal = torch.as_tensor(
        env.random_free_space(include_theta=True), dtype=torch.float32
    )

    # Plot the env, start, and goal
    # env.plot(close=False, display=False)
    # plt.scatter(start[0], start[1], c="green", marker="o", label="Start")
    # plt.scatter(goal[0], goal[1], c="red", marker="x", label="Goal")
    # plt.legend()
    # plt.show()

    goal_reached = False
    trajectory = []
    trajectory.append(start)

    while False and not check_path_collision(env, trajectory, goal):
        # predict next point using most recent point
        next = model(map_tensor, curr_point[2], goal)
        next = next.squeeze(0).cpu().detach().numpy()
        print(f"Next point: {next}")

        # steer to point (fit dubins and verify collision free)
        # add the point to the trajectory

        # try to connect steer point to goal
        # if can, addgoal to trajectory and return

    trajectory.append(goal)

    # Plot the environment
    env.plot(close=False, display=False)
    plt.scatter(start[0], start[1], c="green", marker="o", label="Start")
    plt.scatter(goal[0], goal[1], c="red", marker="x", label="Goal")

    # plot all points on the trajectory as an arrow
    for point in trajectory:
        plt.quiver(
            point[0],
            point[1],
            np.cos(point[2]),
            np.sin(point[2]),
            color="black",
            angles="xy",
            scale_units="xy",
            scale=0.5,
        )

    # plot a simple dubins path for reference
    simple_path, _ = DUBINS.dubins_path(start, goal)
    plt.plot(
        simple_path[:, 0],
        simple_path[:, 1],
        color="blue",
        label="Dubins Path",
    )

    # plot the trajectory, interpolating dubins path between points
    # each interior point of the trajectory should be an arrow

    plt.legend()
    plt.show()

    # could also run RRT* to find a path? And record the timing information? If I end up having time or caring enough


if __name__ == "__main__":
    main()
