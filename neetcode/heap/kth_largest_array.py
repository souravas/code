import heapq


class Solution:
    def findKthLargest(self, nums: list[int], k: int) -> int:
        min_heap = nums
        heapq.heapify(min_heap)
        while len(min_heap) > k:
            heapq.heappop(min_heap)
        return min_heap[0]
