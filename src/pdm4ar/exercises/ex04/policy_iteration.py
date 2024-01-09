from typing import Tuple
from .structures import Cell, Action


import numpy as np

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy
from pdm4ar.exercises_def.ex04.utils import time_function


class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        is_stable = False
        threshold = 1e-4 


        while not is_stable:
            # Policy Evaluation
            while True:
                delta = 0
                for state in np.ndindex(grid_mdp.grid.shape):
                    if grid_mdp.grid[state] == Cell.GOAL:
                        continue
                    old_value = value_func[state]
                    action = Action(policy[state])
                    next_state = grid_mdp.get_next_state(state, Action(policy[state])) # Compute next state based on current policy's action
                    reward = grid_mdp.stage_reward(state, action)
                    value_func[state] = reward + grid_mdp.gamma * value_func[next_state]
                    delta = max(delta, abs(old_value - value_func[state]))
                if delta < threshold:
                    break

            # Policy Improvement
            is_stable = True
            for state in np.ndindex(grid_mdp.grid.shape):
                if grid_mdp.grid[state] == Cell.GOAL:
                    continue
                old_action = policy[state]
                max_value = float('-inf')
                for action in Action:
                    next_state = grid_mdp.get_next_state(state, action)  # Compute the next state based on the action
                    if not grid_mdp.is_valid_transition(state, action, next_state):
                        continue
                    reward = grid_mdp.stage_reward(state, action)
                    value = reward + grid_mdp.gamma * value_func[next_state]
                    if value > max_value:
                        max_value = value
                        policy[state] = int(action)
                if old_action != policy[state]:
                    is_stable = False
        
        return value_func, policy