from typing import Optional

from neetcode.trees.tree_node import TreeNode


class Solution:
    def maxDepth1(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    def maxDepth2(self, root: Optional[TreeNode]) -> int:
        def dfs(root, depth):
            if root is None:
                return depth
            return max(dfs(root.left, depth + 1), dfs(root.right, depth + 1))

        dfs(root, 0)
