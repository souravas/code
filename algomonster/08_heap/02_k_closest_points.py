import heapq


def k_closest_points_min_heap(points: list[list[int]], k: int) -> list[list[int]]:
    heap = []
    for point in points:
        distance = (point[0] ** 2) + (point[1] ** 2)
        heapq.heappush(heap, (distance, point))
    result = []
    for _ in range(k):
        current = heapq.heappop(heap)
        result.append(current[1])

    return result


def k_closest_points_max_heap(points: list[list[int]], k: int) -> list[list[int]]:
    heap = []
    for point in points:
        distance = (point[0] ** 2) + (point[1] ** 2)
        heapq.heappush(heap, (-distance, point))
        if len(heap) > k:
            heapq.heappop(heap)
    result = []
    while heap:
        result.append(heapq.heappop(heap)[1])
    result.reverse()

    return result
