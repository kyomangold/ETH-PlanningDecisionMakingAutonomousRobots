from typing import Sequence

from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftCommands, SpacecraftState
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
'''
# Importing custom modules
from .path_planning import a_star_search, reconstruct_path
from .control_system import PIDController
from .collision_avoidance import calculate_repulsive_force
from .utils import limit_value, rotate_vector
from .config import *
'''
# Importing custom models
from .path_planning import RRTStarPlanner
from .control_system import MPCController
from .utils import smooth_trajectory, calculate_enclosing_circle

import numpy as np

class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do NOT modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    def __init__(self,
                 goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 sg: SpacecraftGeometry,
                 sp: SpacecraftGeometry,
                 start_position=None
                 area):
        self.goal = goal
        self.static_obstacles = static_obstacles
        self.sg = sg
        self.sp = sp
        self.name = None
        
        if start_position is None: 
            start_position = [0, 0]  # Default start position
        if area is None:
            area = [-100, -100, 100, 100]  # Default values
       
        self.rrt_star_planner = RRTStarPlanner(start=sg.start_position, goal=goal, obstacles=static_obstacles, area=area)   
        '''
        # Initialize the PID control systems
        self.linear_pid = PIDController(PID_KP, PID_KI, PID_KD)
        self.angular_pid = PIDController(PID_KP, PID_KI, PID_KD)
        '''
        # Initialize RRT* Planner
        self.rrt_star_planner = RRTStarPlanner(start=sg.start_position, goal=goal, obstacles=static_obstacles, area=area)
        # Initialize MPC Controller
        self.mpc_controller = MPCController(horizon=10, dt=0.1, max_acceleration=10)



    def on_episode_init(self, my_name: PlayerName):
        self.name = my_name

    def get_commands(self, sim_obs: SimObservations) -> SpacecraftCommands:
        """ This method is called by the simulator at each time step.

        This is how you can get your current state from the observations:
        my_current_state: SpacecraftState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """
        
        # Extract the current state of the spacecraft
        my_state = sim_obs.players[self.name].state
        current_state = [my_state.x, my_state.y, my_state.psi, my_state.vx, my_state.vy, my_state.dpsi]

        # Plan the path using RRT*
        path = self.rrt_star_planner.plan_path()
        if path is None:
            print("No path found.")
            return SpacecraftCommands(0, 0)

        # Convert path to the format expected by MPC
        path = np.array(path)
        if path.shape[0] < self.mpc_controller.horizon:
            # Pad the path if it's shorter than the MPC horizon
            last_point = path[-1, :]
            needed_points = self.mpc_controller.horizon - path.shape[0]
            path = np.vstack([path, np.repeat([last_point], needed_points, axis=0)])

        # Smoothing the path
        smooth_path = smooth_trajectory(path, self.sp.max_velocity, (current_x, current_y))

        # Compute control commands using MPC with the smoothed path
        acc_left, acc_right = self.mpc_controller.compute_commands(current_state, smooth_path)


        return SpacecraftCommands(acc_left, acc_right)
        
        '''
        PID Implementation

        my_current_state: SpacecraftState = sim_obs.players[self.name].state
        
        # Define the starting point and goal for the path planning
        start = (my_current_state.position.x, my_current_state.position.y)
        goal = (self.goal.center.x, self.goal.center.y)
        
        # Use A* to find the path
        # Note: A function to find neighbors needs to be implemented based on the specific simulation environment
        came_from, cost_so_far = a_star_search(start, goal, neighbors_func=self.find_neighbors)
        path = reconstruct_path(came_from, start, goal)

        # Calculate repulsive forces for collision avoidance
        obstacles_positions = [(obstacle.position.x, obstacle.position.y) for obstacle in self.static_obstacles]
        repulsive_force = calculate_repulsive_force(my_current_state.position.as_tuple(), obstacles_positions, INFLUENCE_RADIUS, REPULSIVE_COEFFICIENT)

        # Use PID controllers to compute the required accelerations
        # Assume that the path gives us the next immediate point to move towards
        target_position = path[1] if len(path) > 1 else path[0]
        direction_to_target = rotate_vector(target_position, -my_current_state.orientation.z)
        linear_acc = self.linear_pid.update(direction_to_target[0], sim_obs.dt)
        angular_acc = self.angular_pid.update(direction_to_target[1], sim_obs.dt)

        # Apply limits to the accelerations
        linear_acc = limit_value(linear_acc, MIN_SPEED, MAX_SPEED)
        angular_acc = limit_value(angular_acc, MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        # Factor in collision avoidance
        linear_acc += repulsive_force[0]
        angular_acc += repulsive_force[1]

        # Return the commands for the spacecraft
        return SpacecraftCommands(acc_left=linear_acc, acc_right=angular_acc)
        '''


        #return SpacecraftCommands(acc_left=1, acc_right=1)
