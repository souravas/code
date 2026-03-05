from collections import deque


def task_scheduling(tasks: list[str], requirements: list[list[str]]):
    def create_graph():
        graph = {task: [] for task in tasks}
        for a, b in requirements:
            graph[a].append(b)
        return graph

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
        for key, value in indegree.items():
            if value == 0:
                queue.append(key)

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) == len(tasks):
            return result
        return None

    return topological_sort(create_graph())
