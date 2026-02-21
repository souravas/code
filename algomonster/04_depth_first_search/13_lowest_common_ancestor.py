class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lca_on_bst(bst: Node | None, p: int, q: int) -> int:
    if not bst:
        return -1

    if bst.val < p and bst.val < q:
        return lca_on_bst(bst.right, p, q)
    elif bst.val > p and bst.val > q:
        return lca_on_bst(bst.left, p, q)
    else:
        return bst.val
