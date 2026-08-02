from collections import deque


def clone_graph(adj_list: list[list[int]]) -> list[list[int]]:
    if not adj_list:
        return []

    n = len(adj_list)
    # Initialize the result list with empty lists
    result = [[] for _ in range(n)]

    # Use BFS to traverse the graph
    queue = deque([0])  # Start from node 0
    visited = {0}

    while queue:
        node = queue.popleft()
        # Copy neighbors directly since we want to maintain the same node values
        result[node] = adj_list[node].copy()

        # Add unvisited neighbors to queue
        for neighbor in adj_list[node]:
            # Convert 1-based neighbor index to 0-based for traversal
            neighbor_idx = neighbor - 1
            if neighbor_idx < n and neighbor_idx not in visited:
                queue.append(neighbor_idx)
                visited.add(neighbor_idx)

    return result
