def dfs(root, target):
    if not root:
        return None

    if root.val == target:
        return root

    return dfs(root.left, target) or dfs(root.right, target)
