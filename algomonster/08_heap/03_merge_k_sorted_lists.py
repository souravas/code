import heapq


def merge_k_sorted_lists(lists: list[list[int]]) -> list[int]:
    heap = []
    result = []

    for numbers in lists:
        heapq.heappush(heap, (numbers[0], numbers, 0))

    while heap:
        number, numbers, index = heapq.heappop(heap)
        result.append(number)
        index += 1
        if index < len(numbers):
            heapq.heappush(heap, (numbers[index], numbers, index))

    return result
