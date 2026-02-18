class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


def has_cycle(nodes: Node) -> bool:
    fast = nodes
    slow = nodes

    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next  # type: ignore
        if fast == slow:
            return True

    return False
