from collections import deque


def bfs(root):
    queue = deque([root])
    while len(queue):
        node = queue.popleft()
        for child in node.children:
            if ok(child):
                return child
            queue.append(child)
    return None


def ok(child): ...
