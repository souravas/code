class _Node:
    def __init__(self):
        self.children = {}
        self.is_word = False


class WordDictionary:
    def __init__(self, words=()):
        self._root = _Node()
        for word in words:
            self.add(word)

    def add(self, word):
        node = self._root
        for ch in word:
            node = node.children.setdefault(ch, _Node())
            # above code is equivalent to below:
            # if ch not in node.children:
            #     node.children[ch] = _Node()
            # node.children[ch]
        node.is_word = True

    def search(self, pattern):
        def match(node, i):
            if i == len(pattern):
                return node.is_word
            ch = pattern[i]
            if ch == ".":
                return any(match(child, i + 1) for child in node.children.values())
            child = node.children.get(ch)
            return child is not None and match(child, i + 1)

        return match(self._root, 0)


def design_add_and_search_words_data_structure(operations):
    dictionary = WordDictionary()
    results = []
    for op, *args in operations:
        if op == "WordDictionary":
            dictionary = WordDictionary()
        elif op == "addWord":
            dictionary.add(*args)
        elif op == "search":
            results.append(dictionary.search(*args))
    return results
