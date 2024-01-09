# control_system.py
import numpy as np
import cvxpy as cp

class MPCController:
    def __init__(self, horizon=10, dt=0.1, max_acceleration=10):
        self.horizon = horizon
        self.dt = dt
        self.max_acceleration = max_acceleration

    def compute_commands(self, current_state, path):
        # Extract current state
        x, y, psi, vx, vy, dpsi = current_state

        # Define the control and state variables
        acc_left = cp.Variable(self.horizon)
        acc_right = cp.Variable(self.horizon)
        X = cp.Variable(self.horizon + 1)
        Y = cp.Variable(self.horizon + 1)
        Psi = cp.Variable(self.horizon + 1)
        Vx = cp.Variable(self.horizon + 1)
        Vy = cp.Variable(self.horizon + 1)
        dPsi = cp.Variable(self.horizon + 1)

        # Define the constraints
        constraints = [X[0] == x, Y[0] == y, Psi[0] == psi,
                       Vx[0] == vx, Vy[0] == vy, dPsi[0] == dpsi]

        for t in range(self.horizon):
            constraints += [
                X[t + 1] == X[t] + Vx[t] * self.dt,
                Y[t + 1] == Y[t] + Vy[t] * self.dt,
                Psi[t + 1] == Psi[t] + dPsi[t] * self.dt,
                Vx[t + 1] == Vx[t] + ((acc_left[t] + acc_right[t]) / 2) * np.cos(Psi[t]) * self.dt,
                Vy[t + 1] == Vy[t] + ((acc_left[t] + acc_right[t]) / 2) * np.sin(Psi[t]) * self.dt,
                dPsi[t + 1] == dPsi[t] + ((acc_right[t] - acc_left[t]) / 2) * self.dt,
                acc_left[t] >= -self.max_acceleration,
                acc_left[t] <= self.max_acceleration,
                acc_right[t] >= -self.max_acceleration,
                acc_right[t] <= self.max_acceleration,
            ]

        # Define the objective function
        objective = cp.Minimize(cp.sum_squares(cp.vstack([X[1:], Y[1:]]) - path[:self.horizon].T))

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Extract the optimal control commands
        acc_left_optimal = acc_left.value[0]
        acc_right_optimal = acc_right.value[0]

        return acc_left_optimal, acc_right_optimal

# Example usage
# current_state = [x, y, psi, vx, vy, dpsi]
# path = np.array([[x1, y1], [x2, y2], ..., [xn, yn]])
# mpc = MPCController()
# acc_left, acc_right = mpc.compute_commands(current_state, path)


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = 0.0
        self.error_sum = 0.0
        self.last_error = 0.0

    def update(self, current_value, dt):
        error = self.set_point - current_value
        self.error_sum += error * dt
        d_error = (error - self.last_error) / dt

        output = (self.kp * error) + (self.ki * self.error_sum) + (self.kd * d_error)
        self.last_error = error

        return output

    def set_target(self, target):
        self.set_point = target
        self.error_sum = 0.0
        self.last_error = 0.0
