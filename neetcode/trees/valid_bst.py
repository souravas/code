from typing import Optional

from neetcode.trees.tree_node import TreeNode


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def is_valid(root, left, right):
            if root is None:
                return True
            if left >= root.val or right <= root.val:
                return False
            return is_valid(root.left, left, root.val) and is_valid(
                root.right, root.val, right
            )

        return is_valid(root, float("-inf"), float("inf"))
