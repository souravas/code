from collections import Counter
import heapq


def reorganize_string(s: str):
    n = len(s)
    str_count = Counter(s)

    queue = [(-value, key) for key, value in str_count.items()]
    heapq.heapify(queue)
    if -queue[0][0] > (n + 1) // 2:
        return ""
    result = [""] * n
    index = 0

    while queue:
        count, key = heapq.heappop(queue)
        count = -count
        for _ in range(count):
            result[index] = key
            index += 2
            if index >= n:
                index = 1
    return "".join(result)
