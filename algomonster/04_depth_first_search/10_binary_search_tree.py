class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def find(tree, val):
    if not tree:
        return False
    if tree.val == val:
        return True
    elif tree.val < val:
        return find(tree.right, val)
    else:
        return find(tree.left, val)


def insert(tree, val):
    if not tree:
        return Node(val)
    if tree.val < val:
        tree.right = insert(tree.right, val)
    elif tree.val > val:
        tree.left = insert(tree.left, val)
    return tree
