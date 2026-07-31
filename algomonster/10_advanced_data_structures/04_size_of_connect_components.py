class SetCounter:
    def __init__(self):
        self.parent = {}
        self.sizes = {}

    def _ensure(self, x: int) -> None:
        # every element starts as its own root, in a set of size 1
        if x not in self.parent:
            self.parent[x] = x
            self.sizes[x] = 1

    def find(self, x: int) -> int:
        self._ensure(x)
        if self.parent[x] == x:
            return x
        # path compression: point x straight at the root on the way back
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def merge(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        self.parent[rx] = ry
        self.sizes[ry] = self.sizes[rx] + self.sizes[ry]

    def count(self, x: int) -> int:
        return self.sizes[self.find(x)]
