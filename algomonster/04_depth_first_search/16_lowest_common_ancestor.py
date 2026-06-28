class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lca(root: Node | None, node1: Node, node2: Node) -> Node | None:
    if not root:
        return root

    if root in (node1, node2):
        return root

    left = lca(root.left, node1, node2)
    right = lca(root.right, node1, node2)

    if left and right:
        return root

    if left:
        return left

    if right:
        return right

    return None
