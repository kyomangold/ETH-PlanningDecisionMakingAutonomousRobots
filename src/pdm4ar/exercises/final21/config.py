# config.py

import numpy as np

# PID Controller gains
PID_KP = 1.0
PID_KI = 0.1
PID_KD = 0.05

# Spacecraft maximum and minimum speeds and angular velocities
MAX_SPEED = 50.0  # km/h
MIN_SPEED = -50.0  # km/h
MAX_ANGULAR_VELOCITY = 2 * np.pi  # rad/s
MIN_ANGULAR_VELOCITY = -2 * np.pi  # rad/s

# Collision avoidance parameters
INFLUENCE_RADIUS = 10.0  # meters
REPULSIVE_COEFFICIENT = 1.0
