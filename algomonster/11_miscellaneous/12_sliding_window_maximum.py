from collections import deque


def sliding_window_maximum(nums: list[int], k: int) -> list[int]:
    queue = deque()
    result = []

    for index, val in enumerate(nums):
        if queue and queue[0] <= index - k:
            queue.popleft()

        while queue and nums[queue[-1]] <= val:
            queue.pop()

        queue.append(index)

        if index >= k - 1:
            result.append(nums[queue[0]])

    return result
