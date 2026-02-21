import math


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def visible_tree_node(root: Node) -> int:

    def dfs(root, max_val):
        if not root:
            return 0

        total = 0

        if root.val >= max_val:
            max_val = root.val
            total += 1
        total += (dfs(root.left, max_val)) + (dfs(root.right, max_val))

        return total

    return dfs(root, -math.inf)
