class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_balanced1(tree: Node) -> bool:
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


def is_balanced2(tree: Node) -> bool:
    def dfs(root):
        if not root:
            return [True, 0]
        left_balanced, left_val = dfs(root.left)
        right_balanced, right_val = dfs(root.right)
        balanced = left_balanced and right_balanced and (abs(abs(left_val) - abs(right_val)) <= 1)
        return [balanced, 1 + max(left_val, right_val)]
    is_balanced, _ = dfs(tree)
    return is_balanced
