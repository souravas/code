class SameSet:

    def __init__(self):
        self.parent = {}

    def _ensure(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def merge(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        self.parent[rx] = ry

    def find(self, x):
        self._ensure(x)
        if self.parent[x] == x:
            return x
        return self.find(self.parent[x])

    def is_same(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)
