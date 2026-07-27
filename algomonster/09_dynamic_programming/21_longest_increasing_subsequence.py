from functools import cache


def longest_sub_len(nums: list[int]) -> int:
    @cache
    def dfs(index):
        max_length = 1
        for j in range(index):
            if nums[j] < nums[index]:
                max_length = max(max_length, 1 + dfs(j))
        return max_length

    result = 0
    for i in range(len(nums)):
        result = max(result, dfs(i))
    return result


from math import inf
from functools import cache


def longest_sub_len_improved(nums: list[int]) -> int:

    @cache
    def dfs(index, previous):
        if index == len(nums):
            return 0

        current = nums[index]
        take = 0
        if current > previous:
            take = 1 + dfs(index + 1, current)
        skip = dfs(index + 1, previous)
        return max(take, skip)

    return dfs(0, -inf)
