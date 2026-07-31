from typing import List


def longest_path(graph: List[List[int]], node: int, parent: int) -> int:
    max_path = 0
    for child in graph[node]:
        if child != parent:
            max_path = max(max_path, longest_path(graph, child, node) + 1)
    return max_path
