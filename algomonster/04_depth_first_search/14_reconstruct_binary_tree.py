class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def construct_binary_tree(preorder: list[int], inorder: list[int]) -> Node | None:
    index = 0
    inorder_map = { value: index for index, value in enumerate(inorder) }

    def construct(left_index, right_index):
        nonlocal index
        if index >= len(preorder) or (right_index - left_index) < 0:
            return None
        root_value = preorder[index]
        index += 1

        mid_index = inorder_map[root_value]

        root = Node(root_value)
        root.left = construct(left_index, mid_index - 1)
        root.right = construct(mid_index + 1, right_index)
        return root

    return construct(0, len(inorder) - 1)
