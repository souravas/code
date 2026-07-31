class UnionFind:
    def __init__(self):
        self.id = {}

    def find(self, x):
        y = self.id.get(x, x)
        if y != x:
            self.id[x] = y = self.find(y)
        return y

    def union(self, x, y):
        self.id[self.find(x)] = self.find(y)


def umbristan(n: int, breaks: list[list[int]]) -> list[int]:
    # initialize data structures
    dsu = UnionFind()
    ret = []
    breaks.reverse()
    # loop through the reverse list and merge the edges if they are not of the same list
    for a, b in breaks:
        ret.append(n)
        # merging 2 connected components means total number of connected components decreases by 1
        if dsu.find(a) != dsu.find(b):
            dsu.union(a, b)
            n -= 1
    # remember that our answers are in reverse since we started from the end
    ret.reverse()
    return ret
