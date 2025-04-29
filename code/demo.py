import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from dynamic_mpnet_planner import DynamicMPNetPlanner
from dynamic_mpnet import DynamicMPNet
from grid_environment import GridEnvironment
from occupancy_grid_generator import OccupancyGridGenerator
from nonlinear_mpc import NMPCSolver
from dubins import Dubins

import io
import imageio

MAP_DIM = (12, 12)
EMBEDDING_DIM = 32
NUM_OBSTACLES = 4

SIM_STEPS = 250
N = 10


DUBINS = Dubins(radius=2, point_separation=0.1)


def simulate_nmpc(env, path, start, goal):

    path = np.stack(path)

    current_state = np.array(start)
    executed_path = [current_state[:2]]  # Store for visualization

    frames = []

    prev_idx = 0

    for step in range(SIM_STEPS):
        if step % 10 == 0:
            print(f"Step {step}/{SIM_STEPS}")

        # Find the closest point on the path to the current state
        distances = np.linalg.norm(path[prev_idx:, :2] - current_state[:2], axis=1)
        closest_idx = np.argmin(distances) + prev_idx
        prev_idx = closest_idx

        N_trim = min(N, len(path) - closest_idx - 1)

        # Create the NMPC solver
        nmpc_solver = NMPCSolver(
            N=N_trim,
            dt=0.1,
            v_s=1.0,
            Q=np.diag([10.0, 10.0, 1.0]),
            R=0.1,
            Qf=np.diag([2000.0, 2000.0, 20.0]),
            u_min=(-np.pi / 3),
            u_max=(np.pi / 3),
        )

        # Solve the NMPC problem
        xref = np.zeros((3, N_trim + 1))

        # Extract the next N+1 points (or remaining points)
        ref_points = path[closest_idx : closest_idx + N_trim + 1]

        # populate the reference trajectory
        xref[0, :] = ref_points[:, 0]
        xref[1, :] = ref_points[:, 1]

        # Calculate the reference heading angles
        for i in range(N_trim + 1):
            if i < N_trim:
                dx = ref_points[i + 1, 0] - ref_points[i, 0]
                dy = ref_points[i + 1, 1] - ref_points[i, 1]
                xref[2, i] = np.arctan2(dy, dx)
            else:
                xref[2, i] = xref[2, i - 1]

        # Solve the NMPC problem
        X_sol, u_sol = nmpc_solver.solve(current_state, xref)

        # Apply the first control input to update the state
        v = nmpc_solver.v_s
        steering = u_sol[0]  # first control input
        dt = nmpc_solver.dt

        next_state = np.array(
            [
                current_state[0] + v * np.cos(current_state[2]) * dt,
                current_state[1] + v * np.sin(current_state[2]) * dt,
                current_state[2] + steering * dt,
            ]
        )

        # store the position
        executed_path.append(next_state[:2])

        # update the current state
        current_state = next_state

        if np.linalg.norm(current_state[:2] - goal[:2]) < 0.5:
            print("Reached the goal!")
            break

        fig, ax = plt.subplots()
        env.plot(display=False)
        ax.plot(xref[0, :], xref[1, :], "ro", markersize=3, label="Reference Points")
        ax.plot(
            X_sol[0, :], X_sol[1, :], "b-", linewidth=2, label="Predicted Trajectory"
        )
        ax.plot(path[:, 0], path[:, 1], "g-", linewidth=2, label="Reference Path")

        # plot the executed path
        ax.plot(
            [p[0] for p in executed_path],
            [p[1] for p in executed_path],
            "k-",
            linewidth=3,
            label="Executed Path",
        )

        # plot the current pose
        ax.scatter(
            current_state[0],
            current_state[1],
            color="blue",
            s=100,
            marker="o",
            label="Current State",
        )

        # draw the vehicle orientation
        arrow_length = 0.3
        ax.arrow(
            current_state[0],
            current_state[1],
            arrow_length * np.cos(current_state[2]),
            arrow_length * np.sin(current_state[2]),
            head_width=0.1,
            head_length=0.15,
            fc="blue",
            ec="blue",
        )

        # plot the start and goal poses
        ax.scatter(start[0], start[1], color="green", s=100, marker="o", label="Start")
        ax.scatter(goal[0], goal[1], color="red", s=100, marker="o", label="Goal")
        ax.legend(loc="upper left")
        ax.axis("equal")
        ax.grid(True)
        ax.set_title(f"NMPC Dubins Path Tracking - Step {step}")

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    imageio.mimsave("nmpc_simulation.gif", frames, fps=20)


def main():

    # Load in the model
    model = DynamicMPNet(MAP_DIM, EMBEDDING_DIM, debug=False)

    model_save_path = os.path.join(
        os.getcwd(), "code/models/dynamic_mpnet_multistep_trained_2.pth"
    )
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Create the planner
    planner = DynamicMPNetPlanner(model, dubins_radius=1.0)

    # Set the random seed for reproducibility
    # np.random.seed(130)
    # np.random.seed(310)
    # np.random.seed(440)
    # np.random.seed(4234)

    # Generate a random oversized map
    map_generator = OccupancyGridGenerator((24, 24), 16)
    map = map_generator.sample()

    # Create a grid environment object
    grid_env = GridEnvironment(map)

    start = grid_env.random_free_space()
    goal = grid_env.random_free_space()

    start = np.asarray(start)
    goal = np.asarray(goal)

    while np.linalg.norm(start - goal) < 15 or np.linalg.norm(start - goal) > 50:
        goal = grid_env.random_free_space()
        goal = np.asarray(goal)

    # Solve the path
    path, ref_pts = planner.plan(grid_env, start, goal)

    if path is None:
        return 1

    print(f"Found a path between points {start} and {goal}")

    path = np.stack(path)

    simulate_nmpc(grid_env, path, start, goal)

    print(f"Wrote simulation to nmpc_simulation.gif")

    return 0


if __name__ == "__main__":
    print(f"Expected Run Time: <1 minute")

    # Set the random seed for reproducibility
    np.random.seed(1823)
    main()
