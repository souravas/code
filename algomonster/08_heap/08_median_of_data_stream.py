from heapq import heappush, heappop


class MedianOfStream:
    def __init__(self):
        self.max_heap = []
        self.min_heap = []

    def add_number(self, num: float) -> None:
        heappush(self.max_heap, -num)
        heappush(self.min_heap, -heappop(self.max_heap))

        if len(self.min_heap) > (len(self.max_heap)):
            heappush(self.max_heap, -heappop(self.min_heap))

    def get_median(self) -> float:
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2
