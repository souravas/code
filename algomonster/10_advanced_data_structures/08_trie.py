class Trie:
    def __init__(self):
        self.children = {}  # char -> child Trie node
        self.freq = 0  # words whose path passes through this node

    def insert(self, word):
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = Trie()
            node = node.children[char]
            node.freq += 1

    def query(self, prefix):
        node = self
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.freq
