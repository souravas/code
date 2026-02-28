from collections import deque


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def binary_tree_min_depth(root: Node) -> int:
    min_level = 0
    queue = deque([root])

    while queue:
        length = len(queue)

        for _ in range(length):
            node = queue.popleft()
            if (not node.left) and (not node.right):
                return min_level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        min_level += 1
    return min_level
