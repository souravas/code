import heapq


def min_meeting_rooms(intervals: list[list[int]]) -> int:
    intervals.sort(key=lambda x: x[0])

    heap = []
    result = 0

    for start, end in intervals:
        while heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)
        result = max(result, len(heap))

    return result
