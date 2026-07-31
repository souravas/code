class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def house_robber_iii(root: Node) -> int:

    def dfs(root):
        if not root:
            return (0, 0)

        left = dfs(root.left)
        right = dfs(root.right)

        rob = root.val + left[1] + right[1]
        skip = max(left) + max(right)

        return (rob, skip)

    return max(dfs(root))
