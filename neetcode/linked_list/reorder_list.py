from typing import Optional

from neetcode.linked_list.list_node import ListNode


class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head:
            return head
        slow, fast = head, head
        # fast.next.next is used instead of fast.
        # when even elements exist, we want first mid instead of second mid.
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        prev, curr = None, slow
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp

        head1, head2 = head, prev
        while head1 and head2:
            next1 = head1.next
            next2 = head2.next

            head1.next = head2
            head1 = next1

            head2.next = head1
            head2 = next2
