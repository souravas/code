from collections import deque


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def binary_tree_right_side_view(root: Node) -> list[int]:
    result = []
    queue = deque([root])

    while queue:
        length = len(queue)

        result.append(queue[-1].val)
        for _ in range(length):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return result
