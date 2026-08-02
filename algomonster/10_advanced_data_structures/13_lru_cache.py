from collections import OrderedDict


class Node:
    def __init__(self, key: int, val: int) -> None:
        self.key, self.val = key, val
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: dict[int, Node] = {}
        # head.next is the most recently used, tail.prev the least
        self.head, self.tail = Node(0, 0), Node(0, 0)
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node: Node) -> None:
        node.prev.next, node.next.prev = node.next, node.prev

    def _insert(self, node: Node) -> None:
        node.prev, node.next = self.head, self.head.next
        node.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._insert(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        if key in self.cache:
            self._remove(self.cache[key])
        self.cache[key] = node = Node(key, value)
        self._insert(node)
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]


class LRUCacheImproved:
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
