from collections import deque


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def closest_values(bst: Node, x: int, k: int) -> list[int]:
    pred, succ = [], []  # stacks of Nodes

    node = bst
    while node:  # seed both stacks along the search path
        if node.val <= x:
            pred.append(node)
            node = node.right
        else:
            succ.append(node)
            node = node.left

    def advance_pred():  # move to next-smaller value
        n = pred.pop().left
        while n:
            pred.append(n)
            n = n.right

    def advance_succ():  # move to next-larger value
        n = succ.pop().right
        while n:
            succ.append(n)
            n = n.left

    out = deque()
    for _ in range(k):
        if not pred and not succ:
            break  # k larger than the tree
        take_pred = succ and pred and (x - pred[-1].val) <= (succ[-1].val - x)
        if pred and (not succ or take_pred):
            out.appendleft(pred[-1].val)
            advance_pred()
        else:
            out.append(succ[-1].val)
            advance_succ()

    return list(out)
