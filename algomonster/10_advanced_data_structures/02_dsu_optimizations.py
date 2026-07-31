class UnionFind:
    # every element starts as its own root
    def __init__(self, n: int):
        self.parent = list(range(n))

    # find the root, then point x straight at it
    def find(self, x: int) -> int:
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    # merge the clusters containing x and y
    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        self.parent[rx] = ry
