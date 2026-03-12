import heapq


def heap_top_3(arr: list[int]) -> list[int]:
    heapq.heapify(arr)
    result = []

    for _ in range(3):
        result.append(heapq.heappop(arr))
    return result
