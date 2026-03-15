import heapq


def nth_ugly_number(n: int) -> int:
    primes = (2, 3, 5)
    heap = [1]
    used_nums = set([1])

    for _ in range(n - 1):
        value = heapq.heappop(heap)
        for prime in primes:
            new_value = prime * value
            if new_value not in used_nums:
                used_nums.add(new_value)
                heapq.heappush(heap, new_value)
    return heap[0]
