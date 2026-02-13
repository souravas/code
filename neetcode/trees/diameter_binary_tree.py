from typing import Optional

from neetcode.trees.tree_node import TreeNode


class Solution:
    def __init__(self):
        self.diameter = 0

    def depth(self, node: Optional[TreeNode]) -> int:
        if node is None:
            return 0
        left = self.depth(node.left)
        right = self.depth(node.right)
        self.diameter = max(self.diameter, left + right)
        return 1 + max(left, right)

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.depth(root)
        return self.diameter
 