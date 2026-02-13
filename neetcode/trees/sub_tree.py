from typing import Optional

from neetcode.trees.tree_node import TreeNode


class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def isSame(root, subRoot):
            if root is None and subRoot is None:
                return True
            if root is None or subRoot is None:
                return False
            if root.val != subRoot.val:
                return False
            return isSame(root.left, subRoot.left) and isSame(root.right, subRoot.right)

        if root is None:
            return False
        if isSame(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
