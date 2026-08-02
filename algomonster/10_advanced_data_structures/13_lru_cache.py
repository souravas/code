from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # mark as most recently used
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # evict least recently used


def execute(operations: list[list[str]]) -> list[int]:
    lru: LRUCache | None = None
    res: list[int] = []
    for operation in operations:
        name = operation[0]
        if name == "LRUCache":
            lru = LRUCache(int(operation[1]))
        elif name == "get":
            res.append(lru.get(int(operation[1])))
        elif name == "put":
            lru.put(int(operation[1]), int(operation[2]))
    return res
