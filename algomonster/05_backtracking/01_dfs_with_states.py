class Node:
    def __init__(self, val, children=None):
        if children is None:
            children = []
        self.val = val
        self.children = children


def ternary_tree_paths(root: Node) -> list[str]:
    def dfs(root, path):
        path.append(str(root.val))
        if not root.children:
            result.append("->".join(path))
        else:
            for child in root.children:
                if child:
                    dfs(child, path)
        path.pop()

    result = []

    if not root:
        return result

    dfs(root, [])

    return result
