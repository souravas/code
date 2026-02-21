class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def subtree_of_another_tree(root: Node | None, sub_root: Node) -> bool:
    def check_same(root, sub_root):
        if not root and not sub_root:
            return True
        if not root or not sub_root:
            return False

        if root.val != sub_root.val:
            return False

        return check_same(root.left, sub_root.left) and check_same(
            root.right, sub_root.right
        )

    if not root:
        return False

    if check_same(root, sub_root):
        return True

    return subtree_of_another_tree(root.left, sub_root) or subtree_of_another_tree(
        root.right, sub_root
    )
