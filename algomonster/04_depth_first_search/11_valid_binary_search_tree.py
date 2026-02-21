from math import inf


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def valid_bst(root: Node) -> bool:
    def is_valid(root, min_val, max_val):
        if not root:
            return True
        if not (min_val < root.val < max_val):
            return False

        return is_valid(root.left, min_val, root.val) and is_valid(
            root.right, root.val, max_val
        )

    return is_valid(root, -inf, inf)
