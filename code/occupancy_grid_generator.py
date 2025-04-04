import numpy as np


class OccupancyGridGenerator:

    def __init__(self, dimensions: tuple, nb_obstacles):
        """
        Initializes the occupancy grid generator

        This class generated occupancy grids with random obstacles
        The grid is indexed [x, y]
        The center 4 cells are always free

        A 0 is free space, a 1 is an obstacle

        """
        self.x_dim = dimensions[0]
        self.y_dim = dimensions[1]
        self.nb_obstacles = nb_obstacles

        self.obstacle_sizes = [(1, 2), (2, 2), (1, 3), (2, 3)]
        self.obstacle_sizes = [2 * size for size in self.obstacle_sizes]

    def sample(self) -> np.ndarray:
        """
        Generates a nxn occupancy grid with nb_obstacles random obstacles

        It is guarnteed that the center 4 cells are free

        The grid is indexed [x, y]
        """

        grid = np.zeros((self.x_dim, self.y_dim), dtype=int)

        # Generate random obstacles
        for _ in range(self.nb_obstacles):
            # select a random obstacle size
            size = self.obstacle_sizes[np.random.randint(len(self.obstacle_sizes))]

            # Randomly rotate the obstacle
            if np.random.rand() > 0.5:
                size = size[::-1]  # one liner to reverse

            x = np.random.randint(0, self.x_dim - size[0])
            y = np.random.randint(0, self.y_dim - size[1])
            grid[x : x + size[0], y : y + size[1]] = 1

        # Set the center 4 cells to free
        grid[
            self.x_dim // 2 - 1 : self.x_dim // 2 + 1,
            self.y_dim // 2 - 1 : self.y_dim // 2 + 1,
        ] = 0

        return grid


if __name__ == "__main__":
    print("Occupancy Grid Generator!")

    # Read in parameters
    print("Enter the dimensions of the grid x y: ")
    dimensions = tuple(map(int, input().split()))

    print("Enter the number of obstacles: ")
    nb_obstacles = int(input())

    # Create the occupancy grid generator
    og = OccupancyGridGenerator(dimensions, nb_obstacles)

    print("Enter the number of samples to generate: ")
    n_samples = int(input())
    print("Generating samples...")

    for i in range(n_samples):
        grid = og.sample()
        np.save(
            f"code/maps/grid_{i}_{dimensions[0]}x{dimensions[1]}_{nb_obstacles}.npy",
            grid,
        )
