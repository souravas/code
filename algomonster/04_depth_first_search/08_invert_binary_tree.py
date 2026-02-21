class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invert_binary_tree(tree: Node | None) -> Node | None:
    if not tree:
        return tree

    tree.left, tree.right = tree.right, tree.left

    invert_binary_tree(tree.left)
    invert_binary_tree(tree.right)

    return tree
