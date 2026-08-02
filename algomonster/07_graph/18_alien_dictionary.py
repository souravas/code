import heapq


def alien_order(words: list[str]) -> str:

    def find_indegree(graph):
        indegree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                indegree[neighbor] += 1
        return indegree

    def topological_sort(graph):
        result = []
        priority_queue = []

        indegree = find_indegree(graph)
        for node in indegree:
            if indegree[node] == 0:
                heapq.heappush(priority_queue, node)
        while priority_queue:
            node = heapq.heappop(priority_queue)
            result.append(node)
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    heapq.heappush(priority_queue, neighbor)
        for value in indegree.values():
            if value != 0:
                return None
        return result

    graph = {}
    for word in words:
        for c in word:
            if c not in graph:
                graph[c] = []

    previous = words[0]
    for i in range(1, len(words)):
        current = words[i]
        j = 0
        while j < len(previous) and j < len(current):
            if previous[j] != current[j]:
                if current[j] not in graph[previous[j]]:
                    graph[previous[j]].append(current[j])
                break
            j += 1
        previous = current

    result = topological_sort(graph)
    if result is None:
        return ""
    return "".join(result)
