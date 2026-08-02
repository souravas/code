from collections import deque


def task_scheduling_2(
    tasks: list[str], times: list[int], requirements: list[list[str]]
) -> int:

    def get_indegree(graph):
        indegree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                indegree[neighbor] += 1
        return indegree

    def topological_sort(graph, task_times):
        result = 0
        queue = deque()
        distance = {node: 0 for node in graph}
        indegree = get_indegree(graph)

        for node in indegree:
            if indegree[node] == 0:
                queue.append(node)
                distance[node] = task_times[node]
                result = max(result, distance[node])

        while queue:
            parent = queue.popleft()
            for child in graph[parent]:
                indegree[child] -= 1
                distance[child] = max(
                    distance[child], distance[parent] + task_times[child]
                )
                result = max(result, distance[child])
                if indegree[child] == 0:
                    queue.append(child)
        return result

    graph = {}
    task_times = {}
    for i in range(len(tasks)):
        graph[tasks[i]] = []
        task_times[tasks[i]] = times[i]

    for requirement in requirements:
        graph[requirement[0]].append(requirement[1])

    return topological_sort(graph, task_times)
