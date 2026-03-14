from heapq import heappush, heappop


def kth_smallest(matrix: list[list[int]], k: int) -> int:
    heap = []
    for row in matrix:
        heappush(heap, (row[0], row, 0))

    while k > 1:
        k -= 1
        value, row, index = heappop(heap)
        index += 1
        if index < len(matrix[0]):
            heappush(heap, (row[index], row, index))
    return heappop(heap)[0]
