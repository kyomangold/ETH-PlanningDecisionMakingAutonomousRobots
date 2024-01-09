# collision_avoidance.py
import numpy as np

def calculate_repulsive_force(position, obstacles, influence_radius, repulsive_coefficient):
    force = np.array([0.0, 0.0])
    for obstacle in obstacles:
        vec_to_obstacle = np.array(obstacle) - np.array(position)
        distance = np.linalg.norm(vec_to_obstacle)
        if distance < influence_radius:
            force -= repulsive_coefficient * (1/distance - 1/influence_radius) * (vec_to_obstacle / distance**3)
    return force
