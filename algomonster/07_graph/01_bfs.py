from collections import deque


def bfs(root):
    queue = deque([root])
    visited = set([root])

    while queue:
        node = queue.popleft()
        for neighbor in get_neighbors(node):
            if neighbor in visited:
                continue
            queue.append(neighbor)
            visited.add(neighbor)


def get_neighbors(node):
    return ""


def bfs_with_level(root):
    queue = deque([root])
    visited = set([root])
    level = 0

    while queue:
        n = len(queue)
        for _ in range(n):
            node = queue.popleft()
            for neighbor in get_neighbors(node):
                if neighbor in visited:
                    continue
                queue.append(neighbor)
                visited.add(neighbor)
        level += 1
