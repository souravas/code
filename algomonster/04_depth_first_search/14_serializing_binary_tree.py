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


def deserialize_with_iter(s):
    # 1. Guard against empty strings up front
    if not s:
        return None

    def dfs(nodes):
        # 2. Provide a default value (None) to prevent StopIteration
        val = next(nodes, None)

        # 3. Handle the default value just like a null node
        if val is None or val == "x":
            return None

        cur = Node(int(val))
        cur.left = dfs(nodes)
        cur.right = dfs(nodes)
        return cur

    return dfs(iter(s.split()))
