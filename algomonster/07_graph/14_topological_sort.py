from collections import deque


def find_indegree(graph):
    indegree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            indegree[neighbor] += 1
    return indegree


def topological_sort(graph):
    result = []
    queue = deque()
    indegree = find_indegree(graph)
    for node in indegree:
        if indegree[node] == 0:
            queue.append(node)

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(graph) == len(result):
        return result
    else:
        return None
