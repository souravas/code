from collections import deque


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def zig_zag_traversal(root: Node) -> list[list[int]]:
    result = []
    queue = deque([root])
    reverse = False

    while queue:
        length = len(queue)
        current_level = []

        for _ in range(length):
            node = queue.popleft()
            current_level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        if reverse:
            current_level.reverse()
        result.append(current_level[:])
        reverse = not reverse

    return result
