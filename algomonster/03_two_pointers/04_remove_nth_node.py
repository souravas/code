class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


def remove_nth_from_end(head, n):
    dummy = Node(0, next=head)
    slow = fast = dummy

    while n > 0:
        fast = fast.next
        n -= 1

    while fast.next:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next

    return dummy.next
