import math


def subarray_sum_shortest(nums: list[int], target: int) -> int:
    left = 0
    shortest = len(nums) + 1
    current_sum = 0

    for right in range(len(nums)):
        current_sum += nums[right]

        while current_sum >= target:
            shortest = min(shortest, right - left + 1)
            current_sum -= nums[left]
            left += 1

    if shortest > len(nums):
        return 0
    return shortest
