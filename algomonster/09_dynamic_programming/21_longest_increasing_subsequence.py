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
