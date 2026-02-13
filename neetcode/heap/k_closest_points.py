import heapq


class Solution:
    def kClosest(self, points: list[list[int]], k: int) -> list[list[int]]:
        max_heap = []

        for point in points:
            dist = -(point[0] ** 2 + point[1] ** 2)
            heapq.heappush(max_heap, (dist, point))
            if len(max_heap) > k:
                heapq.heappop(max_heap)
        return [point for (dist, point) in max_heap]
