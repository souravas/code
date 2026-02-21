class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_balanced(tree: Node) -> bool:
    def dfs(root):
        nonlocal balanced
        if not balanced or not root:
            return 0

        left_depth = dfs(root.left)
        right_depth = dfs(root.right)

        if abs(left_depth - right_depth) > 1:
            balanced = False

        return max(left_depth, right_depth) + 1

    balanced = True
    dfs(tree)
    return balanced
