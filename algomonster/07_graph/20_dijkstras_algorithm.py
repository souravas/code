from collections import deque
from math import inf
from heapq import heappop, heappush


def shortest_path_naive(graph: list[list[tuple[int, int]]], a: int, b: int):
    def bfs(root, target):
        queue = deque([root])
        distance = [inf] * len(graph)
        distance[root] = 0

        while queue:
            node = queue.popleft()
            for neighbor, weight in graph[node]:
                if distance[neighbor] <= distance[node] + weight:
                    continue
                queue.append(neighbor)
                distance[neighbor] = distance[node] + weight
        return distance[target]

    result = bfs(a, b)
    if result == inf:
        return -1
    return result


def shortest_path_dijkstra(graph: list[list[tuple[int, int]]], a: int, b: int):
    def bfs(root, target):
        queue = []
        distances = []

        for i in range(len(graph)):
            if i == root:
                distances.append(0)
                heappush(queue, (0, i))
            else:
                distances.append(inf)
                heappush(queue, (inf, i))
        while queue:
            distance, node = heappop(queue)
            if distance > distances[node]:
                continue
            for neighbor, weight in graph[node]:
                d = distances[node] + weight
                if distances[neighbor] <= d:
                    continue
                heappush(queue, (d, neighbor))
                distances[neighbor] = d

        return distances[target]

    result = bfs(a, b)
    if result == inf:
        return -1
    return result
