"""
Environment is represented by a grid world
Implements the StaticEnvironment interface
"""

from environment import StaticEnvironment
import matplotlib.pyplot as plt

import numpy as np


class GridEnvironment:
    """
    Simple 2D world, obstacles are represented by an occupancy grid

    0 is free space, 1 is an obstacle
    """

    def __init__(self, grid):
        """
        Initializes the environment
        """
        self.grid = grid
        self.dimensions = grid.shape

    def plot(self, close=False, display=True):
        """
        Creates a figure and plots the environement on it.

        Parameters
        ----------
        close : bool
            If the plot needs to be automatically closed after the drawing.
        display : bool
            If the view pops up or not (used when generating many images)
        """
        plt.ion() if display else plt.ioff()
        plt.imshow(
            self.grid.T,
            origin="lower",
            cmap="gray_r",
            extent=(0, self.dimensions[0], 0, self.dimensions[1]),
            interpolation="nearest",
        )
        plt.gca().set_xlim(0, self.dimensions[0])
        plt.gca().set_ylim(0, self.dimensions[1])
        if close:
            plt.close()

    def is_free(self, x, y, time=0):
        """
        Returns False if a point is within an obstacle or outside of the
        boundaries of the environement.
        """
        x = int(x)
        y = int(y)

        if x < 0 or x >= self.dimensions[0] or y < 0 or y >= self.dimensions[1]:
            return False
        return self.grid[x, y] == 0

    def random_free_space(self, include_theta=True):
        """
        Returns a random free space in the environment

        Parameters
        ----------
        include_theta : bool
            If the theta value should be included in the return value
            This is kept for compatability with the StaticEnvironment interface

        Returns
        -------
        x : float
            The x coordinate of the free space
        y : float
            The y coordinate of the free space
        theta : float
            Random theta value
        """
        x = np.random.rand() * self.dimensions[0]
        y = np.random.rand() * self.dimensions[1]
        while not self.is_free(x, y) or np.isnan(x) or np.isnan(y):
            x = np.random.rand() * self.dimensions[0]
            y = np.random.rand() * self.dimensions[1]

        if include_theta:
            return x, y, np.random.rand() * 2 * np.pi
        else:
            return x, y
