from neetcode.trees.tree_node import TreeNode


class Solution:
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        current = root.val
        if current < p.val and current < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif current > p.val and current > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root
