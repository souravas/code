from typing import Optional

from neetcode.trees.tree_node import TreeNode


class Solution:
    def __init__(self):
        self.index = 0
        self.result = 0

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder_traversal(root):
            if root is None or self.index > k:
                return
            inorder_traversal(root.left)
            self.index += 1
            if self.index == k:
                self.result = root.val
                return
            inorder_traversal(root.right)

        inorder_traversal(root)
        return self.result
