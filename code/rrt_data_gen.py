"""
@file rrt_data_gen.py
@brief Uses the rrt star implementation to generate referernce trajectories

@details Uses the map_generator to create large maps (48x48) with a proportionate number of obstacles
For each map, we then generate 10 full start-end trajectories, with start and endpoints randomized
The generated trajectories and map are then saved to a json file
"""

import numpy as np
from rrt import DubinsRRT
from grid_environment import GridEnvironment
import json
import matplotlib.pyplot as plt


def generate_rrt_trajectory(
    rrt,
    environment: GridEnvironment,
    max_iter: int = 200,
) -> list:
    """
    Generates a trajectory using the RRT* algorithm

    Returns a list of points representing that trajectory, interpolated
    in step_size increments
    """

    start = environment.random_free_space()
    goal = environment.random_free_space()

    # Create the RRT* object
    rrt.set_start(start)

    success = rrt.run(goal, max_iter)

    if not success:
        print("Failed to find a path from start to goal")
        return None

    # Get the path from the RRT* object
    path, length = rrt.get_best_path(goal)

    return path


def choose_goal(map, start: tuple) -> tuple:
    """
    Choose an accessible goal point on the perimeter of the map
    """
    # Get the dimensions of the map

    x_dim, y_dim = map.shape

    # Choose a random point on the perimeter
    perimeter_points = [
        (0, np.random.randint(0, y_dim)),
        (x_dim - 1, np.random.randint(0, y_dim)),
        (np.random.randint(0, x_dim), 0),
        (np.random.randint(0, x_dim), y_dim - 1),
    ]

    sample = np.random.randint(0, len(perimeter_points))
    goal = perimeter_points[sample]

    while map[goal] == 1:
        sample = np.random.randint(0, len(perimeter_points))
        goal = perimeter_points[sample]

    return goal


def save_data(
    map: GridEnvironment,
    trajectories: list,
    filename: str,
) -> None:
    """
    Saves the trajectories and map to a JSON file
    """

    # Create a dictionary to store the data
    data = {
        "map": map.grid.tolist(),  # Convert numpy array to list for JSON serialization
        "trajectories": [trajectory.tolist() for trajectory in trajectories],
    }

    # Save the data to a JSON file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
        print(f"Saved data to {filename}")


import os
from occupancy_grid_generator import OccupancyGridGenerator


def create_rrt_data(
    num_maps: int = 100,
    num_samples_per: int = 10,
    out_directory: str = "code/data",
    map_size: int = 48,
    num_obstacles: int = 16 * 4,
    rrt_iter: int = 100,
    rrt_stepsize: float = 0.5,
    dubins_radius: float = 2.0,
    do_plot: bool = False,
    do_save: bool = True,
) -> None:
    """
    Generate large RRT* sample trajectories, that will be be used for training

    We use a large map size (48x48) with a proportionate number of obstacles
    We generate 10 trajectories per map, with start and endpoints randomized
    The generated trajectories and map are then saved to a json file

    Parameters
    ----------
    num_maps : int
        The number of maps to generate
    num_samples_per : int
        The number of samples to generate per map
    out_directory : str
        The directory to save the generated maps and trajectories
    map_size : int
        The size of the map (map_size x map_size)
    num_obstacles : int
        The number of obstacles to generate in the map
    rrt_iter : int
        The number of iterations for the RRT* algorithm
    rrt_stepsize : float
        The step size for the RRT* algorithm
    dubins_radius : float
        The radius for the Dubins path
    do_plot : bool
        Whether to plot the generated map and trajectories
    do_save : bool
        Whether or not to save the generated data

    Returns
    -------
    None
        The function saves the generated maps and trajectories to a file
    """

    run_id = np.random.randint(100000).__str__()
    save_dir = os.path.join(out_directory, run_id)

    # Create the output directory if it doesn't exist
    if do_save:
        os.makedirs(save_dir, exist_ok=True)

    # Create the occupancy grid generator
    generator = OccupancyGridGenerator((map_size, map_size), num_obstacles)

    save_metadata = {
        "map_size": map_size,
        "num_obstacles": num_obstacles,
        "num_maps": num_maps,
        "rrt_stepsize": rrt_stepsize,
        "dubins_radius": dubins_radius,
        "map_metadata": [],
    }

    # Run the algorithm num_maps times to generate data
    for i in range(num_maps):
        map = GridEnvironment(generator.sample())
        rrt = DubinsRRT(
            map,
            radius=dubins_radius,
            point_separation=rrt_stepsize,
        )

        # Store the trajectories for this map
        map_trajectories = []

        for j in range(num_samples_per):
            trajectory = generate_rrt_trajectory(rrt, map, rrt_iter)

            if trajectory is None:
                print(f"Failed to generate trajectory for map {i} iteration {j}")
                continue
            else:
                map_trajectories.append(trajectory)

        # Plot the map and all generated trajectories
        if do_plot:
            print(f"Plotting Trajectories!")
            map.plot(display=False)
            for j, trajectory in enumerate(map_trajectories):
                plt.plot(trajectory[0][0], trajectory[0][1], "ro")
                plt.plot(trajectory[-1][0], trajectory[-1][1], "go")
                plt.plot(
                    [point[0] for point in trajectory],
                    [point[1] for point in trajectory],
                    label=f"Trajectory {j}",
                )

            plt.legend()
            plt.show()

        # Save the map and all trajectories to a file
        if do_save:
            meta = {
                "num_trajectories": len(map_trajectories),
            }

            save_metadata["map_metadata"].append(meta)

            np.save(
                f"{save_dir}/map_{i}.npy",
                map.grid,
            )
            for j, trajectory in enumerate(map_trajectories):
                np.save(
                    f"{save_dir}/map_{i}_trajectory_{j}.npy",
                    trajectory,
                )

    if do_save:
        try:
            with open(
                f"{save_dir}/metadata.json",
                "w",
            ) as f:
                json.dump(save_metadata, f, indent=4)
                print(f"Saved metadata to {f.name}")
        except Exception as e:
            print(f"Failed to save metadata to {save_dir}/metadata.json")
            print(e)
            return


import sys
import argparse

parser = argparse.ArgumentParser("rrt_data_gen")
parser.add_argument(
    "--num_maps", type=int, default=100, help="Number of maps to generate"
)
parser.add_argument(
    "--num_samples_per", type=int, default=10, help="Number of samples per map"
)
parser.add_argument(
    "--out_directory",
    type=str,
    default="code/data",
    help="Directory to save the generated maps and trajectories",
)
parser.add_argument(
    "--map_size", type=int, default=48, help="Size of the map (map_size x map_size)"
)
parser.add_argument(
    "--num_obstacles",
    type=int,
    default=16 * 4,
    help="Number of obstacles to generate in the map",
)
parser.add_argument(
    "--rrt_iter",
    type=int,
    default=200,
    help="Number of iterations for the RRT* algorithm",
)
parser.add_argument(
    "--rrt_stepsize",
    type=float,
    default=0.5,
    help="Step size for the RRT* algorithm",
)
parser.add_argument(
    "--dubins_radius",
    type=float,
    default=2.0,
    help="Radius for the Dubins path",
)
parser.add_argument(
    "--plot",
    action="store_true",
    help="Plot the generated map and trajectories",
)
parser.add_argument(
    "--no_save",
    action="store_false",
    help="Do not save the generated map and trajectories",
)


def main():
    """
    Used to pass params directly to create_rrt_data()
    """
    args = parser.parse_args()

    create_rrt_data(
        num_maps=args.num_maps,
        num_samples_per=args.num_samples_per,
        out_directory=args.out_directory,
        map_size=args.map_size,
        num_obstacles=args.num_obstacles,
        rrt_iter=args.rrt_iter,
        rrt_stepsize=args.rrt_stepsize,
        dubins_radius=args.dubins_radius,
        do_plot=args.plot,
        do_save=args.no_save,
    )


if __name__ == "__main__":
    main()
