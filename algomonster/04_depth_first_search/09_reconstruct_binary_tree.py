class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def construct_binary_tree(preorder: list[int], inorder: list[int]) -> Node | None:
    preorder_index = 0

    def construct(inorder) -> Node | None:
        if not inorder:
            return None
        nonlocal preorder_index
        current = preorder[preorder_index]
        root = Node(current)
        preorder_index += 1

        partition = -1
        for index, value in enumerate(inorder):
            if value == current:
                partition = index
                break

        left = inorder[:partition]
        right = inorder[partition + 1 :]
        root.left = construct(left)
        root.right = construct(right)

        return root

    return construct(inorder)


def construct_binary_tree_optimized(
    preorder: list[int], inorder: list[int]
) -> Node | None:
    preorder_index = 0

    inorder_map = {value: index for index, value in enumerate(inorder)}

    def construct(left, right) -> Node | None:
        if left > right:
            return None
        nonlocal preorder_index
        current = preorder[preorder_index]
        preorder_index += 1

        root = Node(current)

        partition = inorder_map[current]

        root.left = construct(left, partition - 1)
        root.right = construct(partition + 1, right)

        return root

    return construct(0, len(inorder) - 1)
