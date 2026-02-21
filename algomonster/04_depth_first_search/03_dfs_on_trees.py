class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


indent_per_level = "    "


def pretty_print(node, indent_level):
    if not node:
        return

    current_indent_level = indent_level + indent_per_level
    print(current_indent_level + node.val)
    pretty_print(node.left, current_indent_level)
    pretty_print(node.right, current_indent_level)


import math


def find_max(node):
    if not node:
        return -math.inf

    left_max_val = find_max(node.left)
    right_max_val = find_max(node.right)

    return max(node.val, left_max_val, right_max_val)


max_val = -math.inf


def find_max_with_global(node):
    global max_val
    if not node:
        return

    if node.val > max_val:
        max_val = node.val

    find_max_with_global(node.left)
    find_max_with_global(node.right)


def get_max_val(root):
    find_max_with_global(root)
    return max_val
