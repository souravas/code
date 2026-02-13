from collections import deque


class Solution:
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        output = []
        q = deque()
        # q stores indexes

        l = r = 0

        while r < len(nums):
            # pop smaller values from qr
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            q.append(r)

            # remove left value from window
            if l > q[0]:
                q.popleft()

            if (r + 1) >= k:
                output.append(nums[q[0]])
                l += 1
            r += 1
        return output
