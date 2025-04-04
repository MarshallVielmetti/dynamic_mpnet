from environment import StaticEnvironment
from dubins import Dubins
import matplotlib.pyplot as plt

import numpy as np

THETA_PENALTY = 0  # Penalty for theta difference
ALPHA = 0.9  # Probability of not sampling the goal
K_RRT = 10


def euclidian_distance_squared(p1, p2):
    """
    Calculate the squared Euclidean distance between two points
    """
    # Compute the theta difference, respecting the periodicity
    theta_diff = p1[2] - p2[2]
    if theta_diff > np.pi:
        theta_diff -= 2 * np.pi
    elif theta_diff < -np.pi:
        theta_diff += 2 * np.pi

    return (
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + THETA_PENALTY * (p1[2] - p2[2]) ** 2
    )


class DubinsRRT:
    """
    Class for Dubins RRT Path Planning

    Methods
    -------
    set_start
        Set the start position of the robot

    run
        Run the RRT* algorithm to find a path from start to goal
    """

    def __init__(self, environment: StaticEnvironment, radius=2.0, debug=False):
        self.env = environment
        self.start = None
        self.goal = None
        self.added_goal = False

        self.debug = debug

        # Vertices - tuple (x, y, theta)
        self.vertices = []
        self.cost: list[float] = []  # cost (float)
        self.parent: list[int] = []  # parent (int)

        self.planner = Dubins(radius=radius, point_separation=0.1)

    def set_start(self, start):
        """
        Set the start position of the robot
        """
        self.start = start
        self.goal = None
        self.added_goal = False

        self.vertices = [start]
        self.cost = [0]
        self.parent = [0]

    def run(self, goal=None, num_iterations=100):
        """
        Run the RRT* algorithm for num_iterations
        """
        print(f"Run: Num_iterations = {num_iterations}, Goal = {goal}")

        if self.start is None:
            raise ValueError("Start position is not set")

        if goal is not None:
            self.goal = goal

        for i in range(num_iterations):
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{num_iterations}")
            else:
                self.print_debug(f"Iteration {i + 1}/{num_iterations}")

            # Sample a random point in the environment
            if np.random.rand() < ALPHA or goal is None or self.added_goal:
                x_rand = self.env.random_free_space()
            else:
                x_rand = goal

            self.print_debug(f"Random Point: {x_rand}")

            # Find the nearest node in the tree
            x_nearest_idx = self.find_nearest_node(x_rand)
            x_nearest = self.vertices[x_nearest_idx]

            # Generate a new node by steering towards the random point
            x_new = self.steer(x_nearest, x_rand)
            # x_new = x_rand  # just try to go

            # If the new node is not in the free space, skip it
            if not self.env.is_free(x_new[0], x_new[1]):
                continue

            # If there is not a collision free path from the nearest node to
            # the new node, skip it
            if not self.collision_free(x_nearest, x_new):
                continue

            X_NEAR = self.get_near_nodes(x_new)

            # Add the new node to the set of vertices
            if x_new == self.goal:
                print("ADDED GOAL")
                self.added_goal = True

            self.vertices.append(x_new)
            x_new_idx = len(self.vertices) - 1

            x_min_idx = x_nearest_idx
            c_min = self.cost[x_nearest_idx] + self.cost_between(x_nearest, x_new)

            # Find the best parent for the new node with the minimum cost
            for node_idx in X_NEAR:
                node = self.vertices[node_idx]

                # Check if there is a collision free path from the node
                # to the new node
                if not self.collision_free(node, x_new):
                    continue

                candidate_cost = self.cost[node_idx] + self.cost_between(node, x_new)

                # Check if the candidate cost is less than the current minimum
                if candidate_cost < c_min:
                    x_min_idx = node_idx
                    c_min = candidate_cost

            # Connect the new node to the tree and set cost and path_to
            self.parent.append(x_min_idx)
            self.cost.append(c_min)

            # Attempt to rewire the tree
            for node_idx in X_NEAR:
                node = self.vertices[node_idx]

                # Check if there is a collision free path from the new node
                # to the node
                if not self.collision_free(x_new, node):
                    continue

                candidate_cost = self.cost[x_new_idx] + self.cost_between(x_new, node)

                # Check if the candidate cost is less than the current cost
                if candidate_cost < self.cost[node_idx]:
                    # update the parent of the node
                    self.parent[node_idx] = x_new_idx
                    cost_change = candidate_cost - self.cost[node_idx]
                    self.cost[node_idx] = candidate_cost

                    # Update the cost of the descendants
                    self.update_descendants_cost(node_idx, cost_change)

        # After For Loop Returns, Check if the goal is reached
        if goal is None:
            return

        for vertex in self.vertices:
            if self.cost_between(vertex, self.goal) < 0.3:
                return True

    def find_nearest_node(self, x_rand):
        """
        Find the nearest node in the tree to the random point

        Uses Euclidian distance as approximation
        """
        i_min = 0
        d_min = float("inf")
        for i, v in enumerate(self.vertices):
            # dist = euclidian_distance_squared(v, x_rand)
            # They are the same point
            # if dist < 0.01:
            #     return i

            d = self.planner.dubins_path_length(v, x_rand)
            if d < d_min:
                d_min = d
                i_min = i

        # Return the index of the nearest node
        return i_min

    def steer(self, x_nearest, x_rand) -> tuple:
        """
        Generate a new node by steering towards the random point
        """
        self.print_debug(f"Steer: x_nearest = {x_nearest}, x_rand = {x_rand}")

        # Generate a Dubins path from the nearest node to the random point
        path, path_length = self.planner.dubins_path(x_nearest, x_rand)
        self.print_debug(f"\t Path of Length {path_length}")

        # Sample a point along the path
        sample_idx = min(len(path) - 1, 10)

        self.print_debug(f"\t Sampled Point: {path[sample_idx]}")
        point = path[sample_idx]

        # Point only contains x, y need to add theta
        if (len(path) == 1) or (path_length == 0):
            point = (point[0], point[1], x_nearest[2])
        else:
            # theta is angle between previous point and current point
            theta = np.atan2(
                (point[1] - path[sample_idx - 1][1]),
                (point[0] - path[sample_idx - 1][0]),
            )
            point = (float(point[0]), float(point[1]), float(theta))

        self.print_debug(f"\t New Point: {point}")
        return point

    def collision_free(self, x_from, x_to):
        """
        Check if the dubins shortest path from x_from to x_to is collision free
        """
        path, path_length = self.planner.dubins_path(x_from, x_to)
        for i in range(len(path)):
            if not self.env.is_free(path[i][0], path[i][1]):
                return False

        return True

    def cost_between(self, x_from, x_to) -> float:
        """
        Calculate the cost between two nodes
        """
        return self.planner.dubins_path_length(x_from, x_to)

    def get_near_nodes(self, x_new):
        """
        Return the nearest nodes to the new node

        Parameters
        ----------
        x_new : tuple
            The new node to check

        Returns
        -------
        list
            List of indices of the near nodes
        """
        num_nearest = K_RRT * np.log(len(self.vertices))
        num_nearest = min(int(num_nearest), len(self.vertices))

        self.print_debug("get_near_nodes")
        self.print_debug(f"\tNum Nearest: {num_nearest}")

        # If we have 5 or fewer nodes, return all indices
        if len(self.vertices) <= 5:
            return list(range(len(self.vertices)))

        distances = [
            (i, self.planner.dubins_path_length(v, x_new))
            for i, v in enumerate(self.vertices)
        ]

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Return the indices of the 5 nearest nodes
        near_indices = [i for i, _ in distances[:num_nearest]]

        return near_indices

    def plot(self):
        """
        Plot the tree and the path
        """
        # Plot the environment
        self.env.plot()

        # Plot the vertices
        for vertex in self.vertices:
            plt.plot(vertex[0], vertex[1], "ro", markersize=0.2)

        # Plot the edges
        for i, vertex in enumerate(self.vertices):
            if self.parent[i] != i:
                parent = self.vertices[self.parent[i]]

                path, len = self.planner.dubins_path(parent, vertex)
                plt.plot(path[:, 0], path[:, 1], "b-", linewidth=0.1)

    def get_best_path(self, goal) -> tuple:
        """
        Get the best path to the goal

        Parameters
        ----------
        goal : tuple
            The goal position

        Returns
        -------
        tuple
            list: List of points in the path
            float: Total length of the path
        """
        # Find the nearest node to the goal
        # nearest_node = self.find_nearest_node(goal)
        # print(f"Nearest Node: {nearest_node}")
        min_dist = float("inf")
        min_i = -1
        for i, vertex in enumerate(self.vertices):
            if euclidian_distance_squared(vertex, goal) < min_dist:
                min_i = i
                min_dist = euclidian_distance_squared(vertex, goal)

        nearest_node = min_i

        # Backtrack to find the path
        path = []
        current_node = nearest_node

        while current_node != self.parent[current_node]:
            path.append(self.vertices[current_node])
            current_node = self.parent[current_node]

        path.append(self.vertices[current_node])
        path.reverse()

        total_length = 0
        for i in range(len(path) - 1):
            total_length += self.planner.dubins_path_length(path[i], path[i + 1])

        # Now calculate the dubins path between every point in the path
        path = [
            self.planner.dubins_path(path[i], path[i + 1])[0]
            for i in range(len(path) - 1)
        ]

        if path == []:
            return [], 0

        # Path is now a python list of numpy arrays
        path = np.concatenate(path, axis=0)

        return path, total_length

    def print_debug(self, message):
        """
        Print debug messages if debug mode is enabled
        """
        if self.debug:
            print(message)

    def update_descendants_cost(self, node_idx, cost_change):
        """
        Update the cost of the descendants of a node
        """
        for i in range(len(self.parent)):
            if self.parent[i] == node_idx:
                self.cost[i] += cost_change
                self.update_descendants_cost(i, cost_change)
