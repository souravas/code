from typing import Optional

from neetcode.linked_list.list_node import ListNode


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode()
        result = head
        carry = 0

        while l1 or l2:
            digit = carry
            if l1:
                digit += l1.val
                l1 = l1.next
            if l2:
                digit += l2.val
                l2 = l2.next
            carry = digit // 10
            digit %= 10
            head.next = ListNode(digit)
            head = head.next
        if carry != 0:
            head.next = ListNode(carry)
        return result.next
