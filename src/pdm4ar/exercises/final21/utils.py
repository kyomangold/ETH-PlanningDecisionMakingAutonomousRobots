# utils.py
import numpy as np
import math, random
from scipy import interpolate

def calculate_enclosing_circle(points):
    """
    Find the smallest circle enclosing all given points.
    
    :param points: A list of points (each point is a tuple (x, y)).
    :return: (x, y, radius) of the smallest enclosing circle.
    """
    def is_point_in_circle(circle, p):
        x, y, r = circle
        return math.hypot(x - p[0], y - p[1]) <= r

    def create_circle_two_points(p1, p2):
        cx = (p1[0] + p2[0]) / 2
        cy = (p1[1] + p2[1]) / 2
        r = math.hypot(p1[0] - cx, p1[1] - cy)
        return cx, cy, r

    def create_circle_three_points(p1, p2, p3):
        A = p2[0] - p1[0]
        B = p2[1] - p1[1]
        C = p3[0] - p1[0]
        D = p3[1] - p1[1]
        E = A * (p1[0] + p2[0]) + B * (p1[1] + p2[1])
        F = C * (p1[0] + p3[0]) + D * (p1[1] + p3[1])
        G = 2 * (A * (p3[1] - p2[1]) - B * (p3[0] - p2[0]))

        if G == 0:  # Collinear points
            return None

        # Center of the circle
        cx = (D * E - B * F) / G
        cy = (A * F - C * E) / G

        # Radius of the circle
        r = math.sqrt((p1[0] - cx)**2 + (p1[1] - cy)**2)

        return cx, cy, r


    def _create_circle(points, p):
        c = (p[0], p[1], 0.0)
        for q in points:
            if not is_point_in_circle(c, q):
                if c[2] == 0.0:
                    c = create_circle_two_points(p, q)
                else:
                    c = create_circle_three_points(p, q, next(pnt for pnt in points if pnt != q))
        return c

    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    circle = None
    for i, p in enumerate(shuffled):
        if circle is None or not is_point_in_circle(circle, p):
            circle = _create_circle(shuffled[:i + 1], p)

    return circle


def smooth_trajectory(path, max_velocity, current_position):
    def find_next_idx(smoothed_path, start_position):
        closest_dist = float('inf')
        closest_idx = 0
        for i, point in enumerate(smoothed_path):
            dist = np.hypot(point[0] - start_position[0], point[1] - start_position[1])
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        return closest_idx

    segmented_path = path
    path_length = sum([np.hypot(segmented_path[i+1][0] - p[0], segmented_path[i+1][1] - p[1]) for i, p in enumerate(segmented_path[:-1])])
    num_points = int(2 * path_length / (max_velocity - 2) * 10)

    order_curve = min(max(len(segmented_path) - 1, 1), 4)
    x_coords, y_coords = zip(*segmented_path)

    tck, *rest = interpolate.splprep([x_coords, y_coords], k=order_curve)
    t = np.linspace(0, 1, num_points)
    interpolated_path = interpolate.splev(t, tck)

    smoothed_path = list(zip(interpolated_path[0], interpolated_path[1]))
    idx_smoothed = find_next_idx(smoothed_path, current_position)
    smoothed_path = smoothed_path[idx_smoothed:]

    return smoothed_path

def rotate_vector(vec, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation_matrix, vec)

def limit_value(value, min_value, max_value):
    return max(min_value, min(max_value, value))
