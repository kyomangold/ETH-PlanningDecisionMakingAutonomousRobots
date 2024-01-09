# path_planning.py
import numpy as np
from queue import PriorityQueue

import numpy as np
import random
from shapely.geometry import Point, LineString, Polygon

''' RRT* implementation '''
class Node:
    def __init__(self, point):
        self.point = point
        self.parent = None
        self.cost = 0.0

class RRTStarPlanner:
    def __init__(self, start, goal, obstacles, area, step_size=5, search_radius=10, max_iterations=1000):
        self.start = Node(Point(start))
        self.goal = Node(Point(goal))
        self.obstacles = obstacles
        self.area = area  # [min_x, min_y, max_x, max_y]
        self.step_size = step_size
        self.search_radius = search_radius
        self.max_iterations = max_iterations
        self.nodes = [self.start]

    def plan_path(self):
        for i in range(self.max_iterations):
            random_node = self.get_random_node()
            nearest_node = self.get_nearest_node(random_node)
            new_node = self.steer(nearest_node, random_node)

            if self.is_collision_free(nearest_node, new_node):
                near_nodes = self.get_near_nodes(new_node)
                best_parent = self.choose_best_parent(new_node, near_nodes)
                if best_parent:
                    self.rewire(new_node, near_nodes)
                    self.nodes.append(new_node)

            if new_node.point.distance(self.goal.point) <= self.step_size:
                return self.generate_final_course(new_node)
            
        # After path is found, possibly apply smoothing
        smoothed_path = smooth_trajectory(self.generate_final_course(self.goal), max_velocity, start_position)
        return smoothed_path

    def get_random_node(self):
        if random.randint(0, 100) > 50:
            return Node(Point(self.goal.point.x, self.goal.point.y))
        else:
            return Node(Point(random.uniform(self.area[0], self.area[2]), random.uniform(self.area[1], self.area[3])))

    def get_nearest_node(self, random_node):
        distances = [node.point.distance(random_node.point) for node in self.nodes]
        nearest_index = distances.index(min(distances))
        return self.nodes[nearest_index]

    def steer(self, nearest_node, random_node):
        direction = np.arctan2(random_node.point.y - nearest_node.point.y, random_node.point.x - nearest_node.point.x)
        new_point = Point(nearest_node.point.x + self.step_size * np.cos(direction), nearest_node.point.y + self.step_size * np.sin(direction))
        new_node = Node(new_point)
        new_node.cost = nearest_node.cost + self.step_size
        new_node.parent = nearest_node
        return new_node

    def is_collision_free(self, node1, node2):
        line = LineString([node1.point, node2.point])
        for obstacle in self.obstacles:
            if line.intersects(obstacle):
                return False
        return True

    def get_near_nodes(self, new_node):
        near_nodes = []
        for node in self.nodes:
            if new_node.point.distance(node.point) <= self.search_radius:
                near_nodes.append(node)
        return near_nodes

    def choose_best_parent(self, new_node, near_nodes):
        if not near_nodes:
            return None

        costs = []
        for node in near_nodes:
            if self.is_collision_free(node, new_node):
                costs.append(node.cost + node.point.distance(new_node.point))

        if not costs:
            return None

        min_cost = min(costs)
        min_index = costs.index(min_cost)
        new_node.cost = min_cost
        return near_nodes[min_index]

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            if self.is_collision_free(node, new_node):
                new_cost = new_node.cost + new_node.point.distance(node.point)
                if node.cost > new_cost:
                    node.parent = new_node
                    node.cost = new_cost

    def generate_final_course(self, goal_node):
        path = []
        while goal_node.parent is not None:
            path.append([goal_node.point.x, goal_node.point.y])
            goal_node = goal_node.parent
        path.append([self.start.point.x, self.start.point.y])
        return path[::-1]

# Example usage
# start = (0, 0)
# goal = (10, 10)
# obstacles = [Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]), Polygon([(5, 5), (

''' A* implementation '''

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def a_star_search(start, goal, neighbors_func):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal:
            break

        for next in neighbors_func(current):
            new_cost = cost_so_far[current] + heuristic(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put((priority, next))
                came_from[next] = current

    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path
