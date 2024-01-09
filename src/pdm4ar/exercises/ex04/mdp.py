from abc import ABC, abstractmethod
from typing import Tuple
from .structures import Cell, Action


import numpy as np
from numpy.typing import NDArray

from pdm4ar.exercises.ex04.structures import Action, Policy, ValueFunc, State


class GridMdp:
    def __init__(self, grid: NDArray[np.int], gamma: float = 0.7):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # Actions are deterministic, so the transition probability is 1 if the action leads to the next state, otherwise 0
        if self.is_valid_transition(state, action, next_state):
            return 1.0
        else:
            return 0.0

    def stage_reward(self, state: State, action: Action) -> float:
         # Rewards depend on the type of cell. The START cell is considered as GRASS.
        cell_type = self.grid[state]
        if cell_type == Cell.GOAL:
            return 10
        elif cell_type == Cell.GRASS or cell_type == Cell.START:
            return -1
        elif cell_type == Cell.SWAMP:
            return -10
        else:
            raise ValueError(f"Unknown cell type {cell_type}")
        
    def is_valid_transition(self, state: State, action: Action, next_state: State) -> bool:
        # This method checks if the transition from state to next_state is valid given the action
        max_row_index = self.grid.shape[0] - 1
        max_col_index = self.grid.shape[1] - 1
        current_row, current_col = state

        # Based on the action, determine what the next state should be
        if action == Action.NORTH and current_row > 0:
            expected_next_state = (current_row - 1, current_col)
        elif action == Action.SOUTH and current_row < max_row_index:
            expected_next_state = (current_row + 1, current_col)
        elif action == Action.EAST and current_col < max_col_index:
            expected_next_state = (current_row, current_col + 1)
        elif action == Action.WEST and current_col > 0:
            expected_next_state = (current_row, current_col - 1)
        elif action == Action.STAY:
            expected_next_state = (current_row, current_col)
        else:
            # If the action would lead outside the grid, it's not a valid transition
            return False

        return next_state == expected_next_state
    
    def get_next_state(self, state: State, action: Action) -> State:
        """Calculate the next state given a current state and action."""
        # Assuming a grid where actions move you one cell in the direction of the action
        # and the grid is a 2D numpy array with indices [row, column]
        delta = {
            Action.NORTH: (-1, 0),
            Action.SOUTH: (1, 0),
            Action.WEST: (0, -1),
            Action.EAST: (0, 1),
            Action.STAY: (0, 0),
        }
        next_state = (state[0] + delta[action][0], state[1] + delta[action][1])
        
        # Check if the new position is within the grid boundaries
        if (0 <= next_state[0] < self.grid.shape[0]) and (0 <= next_state[1] < self.grid.shape[1]):
            return next_state
        else:
            # If it's not a valid transition, return the original state
            return state


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        pass
