from functools import cache


def find_largest_subset(nums: list[int]) -> int:
    @cache
    def dfs(i):
        if i == 0:
            return 1
        max_len = 1
        for j in range(i):
            if nums[i] % nums[j] == 0:
                max_len = max(max_len, 1 + dfs(j))
        return max_len

    nums.sort()
    result = 0
    for i in range(len(nums)):
        result = max(result, dfs(i))
    return result
