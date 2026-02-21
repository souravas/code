class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def in_order_traversal(root: Node) -> None:
    if root:
        in_order_traversal(root.left)
        print(root.val)
        in_order_traversal(root.right)


def pre_order_traversal(root: Node) -> None:
    if root:
        print(root.val)
        pre_order_traversal(root.left)
        pre_order_traversal(root.right)


def post_order_traversal(root: None) -> None:
    if root:
        post_order_traversal(root.left)
        post_order_traversal(root.right)
        print(root.val)
