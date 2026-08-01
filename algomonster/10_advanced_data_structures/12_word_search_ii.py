class _Node:
    def __init__(self):
        self.children = {}
        self.word = None  # the full word, if one ends here


class Trie:
    def __init__(self, words=()):
        self.root = _Node()
        for word in words:
            self.add(word)

    def add(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = _Node()
            node = node.children[ch]
        node.word = word


def word_search_ii(matrix: list[str], words: list[str]) -> list[str]:
    if not matrix or not matrix[0]:
        return []

    trie = Trie(words)
    rows, cols = len(matrix), len(matrix[0])
    grid = [list(row) for row in matrix]
    found: set[str] = set()

    def dfs(r, c, node):
        ch = grid[r][c]
        child = node.children.get(ch)
        if child is None:
            return

        if child.word is not None:
            found.add(child.word)
            child.word = None  # don't re-find it later

        grid[r][c] = "#"
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#":
                dfs(nr, nc, child)
        grid[r][c] = ch

        if not child.children:  # dead end, prune it
            del node.children[ch]

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie.root)

    return [word for word in words if word in found]
