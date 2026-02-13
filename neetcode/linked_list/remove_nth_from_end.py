from typing import Optional

from neetcode.linked_list.list_node import ListNode


class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        slow, fast = dummy, dummy
        while n > 0:
            fast = fast.next
            n -= 1
        while fast and fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next
