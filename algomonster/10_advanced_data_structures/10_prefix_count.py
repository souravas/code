class _Node:
    def __init__(self):
        self.freq = 0
        self.children = {}


class Trie:
    def __init__(self, words=()):
        self._root = _Node()
        for word in words:
            self.insert(word)

    def insert(self, word):
        node = self._root
        for ch in word:
            node = node.children.setdefault(ch, _Node())
            node.freq += 1

    def count(self, prefix):
        node = self._root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return 0
        return node.freq


def prefix_count(words: list[str], prefixes: list[str]) -> list[int]:
    trie = Trie(words)
    return [trie.count(p) for p in prefixes]