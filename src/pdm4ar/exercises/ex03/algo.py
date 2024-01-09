from abc import ABC, abstractmethod
from typing import Optional, List
from queue import PriorityQueue
from math import sqrt

from pdm4ar.exercises.ex02.structures import X
from pdm4ar.exercises.ex03.structures import WeightedGraph


class InformedGraphSearch(ABC):
    @abstractmethod
    def path(self, graph: WeightedGraph, start: X, goal: X) -> Optional[List[X]]:
        # need to introduce weights!
        pass
    @abstractmethod
    def heuristic(self, graph: WeightedGraph, current: X, goal: X) -> float:
        pass

    def reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from.get(current)
            if current is None:
                return None
        path.append(start)
        path.reverse()
        return path


class UniformCostSearch(InformedGraphSearch):
    def path(self, graph: WeightedGraph, start: X, goal: X) -> Optional[List[X]]:
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not frontier.empty():
            current_cost, current = frontier.get()

            if current == goal:
                break

            for next in graph.adj_list[current]:
                new_cost = current_cost + graph.get_weight(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost
                    frontier.put((priority, next))
                    came_from[next] = current

        return self.reconstruct_path(came_from, start, goal)
    
     # Uniform Cost Search does not use a heuristic function.
    def heuristic(self, graph: WeightedGraph, current: X, goal: X) -> float:
        return 0


class GreedyBestFirst(InformedGraphSearch):
    def path(self, graph: WeightedGraph, start: X, goal: X) -> Optional[List[X]]:
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not frontier.empty():
            _, current = frontier.get()

            if current == goal:
                break

            for next in graph.adj_list[current]:
                if next not in came_from:
                    priority = self.heuristic(graph, next, goal)
                    frontier.put((priority, next))
                    came_from[next] = current

        return self.reconstruct_path(came_from, start, goal)
    
    def heuristic(self, graph: WeightedGraph, current: X, goal: X) -> float:
        current_x, current_y = graph.get_node_attribute(current, "x"), graph.get_node_attribute(current, "y")
        goal_x, goal_y = graph.get_node_attribute(goal, "x"), graph.get_node_attribute(goal, "y")
        return sqrt((current_x - goal_x) ** 2 + (current_y - goal_y) ** 2)

class Astar(InformedGraphSearch):
    def path(self, graph: WeightedGraph, start: X, goal: X) -> Optional[List[X]]:
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not frontier.empty():
            current_cost, current = frontier.get()

            if current == goal:
                break

            for next in graph.adj_list[current]:
                new_cost = cost_so_far[current] + graph.get_weight(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(graph, next, goal)
                    frontier.put((priority, next))
                    came_from[next] = current

        return self.reconstruct_path(came_from, start, goal)
    
    def heuristic(self, graph: WeightedGraph, current: X, goal: X) -> float:
        # Use the Euclidean distance as the heuristic for A*
        current_x, current_y = graph.get_node_attribute(current, "x"), graph.get_node_attribute(current, "y")
        goal_x, goal_y = graph.get_node_attribute(goal, "x"), graph.get_node_attribute(goal, "y")
        return sqrt((current_x - goal_x) ** 2 + (current_y - goal_y) ** 2)



def compute_path_cost(wG: WeightedGraph, path: List[X]):
    """A utility function to compute the cumulative cost along a path"""
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
