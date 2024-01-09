from typing import Tuple
from .structures import Cell, Action
import numpy as np

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid, dtype=np.float)
        
        # Initialize the policy with a default action
        default_action = Action.STAY.value  # Assuming that Policy expects the integer value of the enum
        policy = np.full_like(grid_mdp.grid, fill_value=default_action, dtype=np.int)


        threshold = 1e-4
        delta = float('inf')

        while delta > threshold:
            delta = 0
            for state in np.ndindex(grid_mdp.grid.shape):
                if grid_mdp.grid[state] == Cell.GOAL:
                    continue  # Skip the goal state

                max_value = float('-inf')
                best_action = Action.STAY
                for action in Action:
                    next_state = grid_mdp.get_next_state(state, action)
                    if not grid_mdp.is_valid_transition(state, action, next_state):
                        continue  # Skip invalid actions

                    reward = grid_mdp.stage_reward(state, action)
                    value = reward + grid_mdp.gamma * value_func[next_state]
                    if value > max_value:
                        max_value = value
                        best_action = action

                delta = max(delta, abs(max_value - value_func[state]))
                value_func[state] = max_value
                policy[state] = best_action.value  # Store the value (integer) of the action enum

        return value_func, policy