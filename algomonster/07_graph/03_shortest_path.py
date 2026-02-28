from collections import deque


def shortest_path(graph: list[list[int]], a: int, b: int) -> int:
    queue = deque([a])
    visited = set([a])
    level = 0

    while queue:
        n = len(queue)

        for _ in range(n):
            node = queue.popleft()
            if node == b:
                return level
            for neighbor in graph[node]:
                if neighbor in visited:
                    continue
                queue.append(neighbor)
                visited.add(neighbor)
        level += 1

    return -1
