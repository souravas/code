class Node:
    def __init__(self, key: int, val: int):
        self.key, self.val = key, val
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}
        # head.next is the most recently used, tail.prev the least
        self.head, self.tail = Node(0, 0), Node(0, 0)
        self.head.next, self.tail.prev = self.tail, self.head

    def remove(self, node: Node) -> None:
        node.prev.next, node.next.prev = node.next, node.prev

    def insert(self, node: Node) -> None:
        node.prev, node.next = self.head, self.head.next
        node.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self.remove(node)
        self.insert(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = node = Node(key, value)
        self.insert(node)
        if len(self.cache) > self.cap:
            lru = self.tail.prev
            self.remove(lru)
            del self.cache[lru.key]


class LRUCache2:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # dicts keep insertion order, so re-inserting marks most recently used
        self.cache[key] = self.cache.pop(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache.pop(key, None)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            del self.cache[next(iter(self.cache))]
