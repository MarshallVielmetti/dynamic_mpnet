import numpy as np
import casadi as ca


# N = 20  # prediction horizon

# Q = np.diag([10.0, 10.0, 1.0])  # state diff weight
# R = 1.0  # control input weight
# Qf = np.diag([20.0, 20.0, 5.0])  # terminal weight

# u_min = -np.pi / 6.0
# u_max = np.pi / 6.0

# v_s = 1.0
# dt = 0.1  # time step


class NMPCSolver:
    def __init__(
        self,
        N=20,
        dt=0.1,
        v_s=1.0,
        Q=np.diag([10.0, 10.0, 1.0]),
        R=0.1,
        Qf=np.diag([2000.0, 2000.0, 20.0]),
        u_min=(-np.pi / 6),
        u_max=(np.pi / 6),
    ):
        self.N = N  # prediction horizon
        self.Q = Q  # state diff weight
        self.R = R  # control input weight
        self.Qf = Qf  # terminal weight
        self.u_min = u_min  # min control input
        self.u_max = u_max  # max control input
        self.v_s = v_s  # speed
        self.dt = dt  # time step

    def solve(self, x0, path):
        """
        Solves the NMPC problem using CasADi and IPOPT

        Parameters
        ----------
        x0 : np.ndarray
            The initial state of the system
        path : np.ndarray
            The desired path to follow (shape: [3, N+1])
        """
        print("Solving NMPC...")
        opti = ca.Opti()

        X = opti.variable(3, self.N + 1)
        U = opti.variable(1, self.N)

        X0 = opti.parameter(3)

        cost = 0
        for k in range(self.N):
            state_error = X[:, k] - path[:, k]
            control_cost = U[:, k]

            cost += ca.mtimes([state_error.T, self.Q, state_error]) + ca.mtimes(
                [control_cost.T, self.R, control_cost]
            )

        # terminal cost
        terminal_state_error = X[:, self.N] - path[:, self.N]
        cost += ca.mtimes([terminal_state_error.T, self.Qf, terminal_state_error])

        opti.minimize(cost)

        # dynamics constraints
        for k in range(self.N):
            x_next = X[0, k] + self.v_s * ca.cos(X[2, k]) * self.dt
            y_next = X[1, k] + self.v_s * ca.sin(X[2, k]) * self.dt
            theta_next = X[2, k] + U[0, k] * self.dt

            opti.subject_to(X[0, k + 1] == x_next)
            opti.subject_to(X[1, k + 1] == y_next)
            opti.subject_to(X[2, k + 1] == theta_next)

        # control input constraints
        opti.subject_to(opti.bounded(self.u_min, U, self.u_max))

        # initial condition
        opti.subject_to(X[:, 0] == X0)

        # solver options
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)
        opti.set_value(X0, x0)

        sol = opti.solve()

        x_sol = sol.value(X)
        U_sol = sol.value(U)

        return x_sol, U_sol
