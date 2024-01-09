from abc import abstractmethod, ABC
from typing import List, Optional, Set, Deque
from collections import deque


from pdm4ar.exercises.ex02.structures import AdjacencyList, X


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        visited = set()
        return self._dfs(graph, start, goal, visited, path=[])

    def _dfs(self, graph: AdjacencyList, current: X, goal: X, visited: Set[X], path: List[X]) -> Optional[List[X]]:
        if current == goal:
            return path + [current]
        visited.add(current)
        for neighbor in graph[current]:
            if neighbor not in visited:
                result = self._dfs(graph, neighbor, goal, visited, path + [current])
                if result is not None:
                    return result
        return None


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        queue: Deque[Tuple[X, List[X]]] = deque([(start, [])])
        visited = set()
        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path + [current]
            visited.add(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [current]))
        return None


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        depth = 0
        while True:
            result = self._dls(graph, start, goal, depth)
            if result is not None:
                return result
            depth += 1

    def _dls(self, graph: AdjacencyList, current: X, goal: X, limit: int, path: List[X] = []) -> Optional[List[X]]:
        if current == goal:
            return path + [current]
        if limit <= 0:
            return None
        for neighbor in graph[current]:
            if neighbor not in path:  # Avoid cycles
                result = self._dls(graph, neighbor, goal, limit - 1, path + [current])
                if result is not None:
                    return result
        return None
