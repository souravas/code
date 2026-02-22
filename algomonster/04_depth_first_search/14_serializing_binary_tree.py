class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def serialize(root):
    result = []

    def dfs(root):
        if not root:
            result.append("x")
            return
        result.append(root.val)
        dfs(root.left)
        dfs(root.right)

    dfs(root)
    return " ".join(result)


def deserialize(s):
    result = s.split()
    index = 0

    def dfs():
        nonlocal index
        if index >= len(result):
            return
        current = result[index]
        index += 1

        if current == "x":
            return None
        current_node = Node(int(current))
        current_node.left = dfs()
        current_node.right = dfs()
        return current_node

    return dfs()
