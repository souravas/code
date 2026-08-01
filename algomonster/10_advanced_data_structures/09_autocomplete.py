class Node:
    def __init__(self):
        self.count = 0  # words inserted so far through this prefix
        self.children = {}


def autocomplete(words: list[str]) -> int:
    root, total = Node(), 0
    for word in words:
        node, typed, shared = root, 0, True
        for ch in word:
            shared = shared and node.count > 0  # some earlier word has this prefix
            typed += shared
            node.count += 1
            node = node.children.setdefault(ch, Node())
        node.count += 1
        total += typed
    return total + 1
