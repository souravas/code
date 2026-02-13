from typing import Optional

from neetcode.trees.tree_node import TreeNode


class Solution:
    def __init__(self):
        self.cache = {}

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def depth(node: Optional[TreeNode]) -> int:
            if node in self.cache:
                return self.cache[node]
            if node is None:
                return 0
            self.cache[node] = 1 + (max(depth(node.left), depth(node.right)))
            return self.cache[node]

        if root is None:
            return True
        return (
            abs(depth(root.left) - depth(root.right)) < 2
            and self.isBalanced(root.left)
            and self.isBalanced(root.right)
        )
