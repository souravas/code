import heapq


class Solution:
    def lastStoneWeight(self, stones: list[int]) -> int:
        max_heap = [-stone for stone in stones]
        heapq.heapify(max_heap)
        while len(max_heap) > 1:
            heapq.heappush(max_heap, heapq.heappop(max_heap) - heapq.heappop(max_heap))
        return -max_heap[0]
