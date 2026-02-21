class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def tree_max_depth(root: Node) -> int:
    if not root:
        return 0

    if root.left is None and root.right is None:
        return 0

    return 1 + max(tree_max_depth(root.left), tree_max_depth(root.right))
