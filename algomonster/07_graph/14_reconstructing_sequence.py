from collections import deque


def sequence_reconstruction(original: list[int], seqs: list[list[int]]) -> bool:
    def find_indegree(graph):
        indegree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                indegree[neighbor] += 1
        return indegree

    def topological_sort(graph):
        sequence = []
        queue = deque()
        indegree = find_indegree(graph)
        for node in indegree:
            if indegree[node] == 0:
                queue.append(node)

        while queue:
            if len(queue) > 1:
                return False

            node = queue.popleft()
            sequence.append(node)
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        return sequence == original

    n = len(original)
    graph = {node: set() for node in range(1, 1 + n)}
    for seq in seqs:
        for i in range(len(seq) - 1):
            source, destination = seq[i], seq[i + 1]
            graph[source].add(destination)
    return topological_sort(graph)
