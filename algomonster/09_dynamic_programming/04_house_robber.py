def rob(nums: list[int]) -> int:
    def dfs(index):
        if index < 0:
            return 0
        if index in memo:
            return memo[index]
        take = nums[index] + dfs(index - 2)
        skip = dfs(index - 1)

        memo[index] = max(take, skip)
        return memo[index]

    memo = {}
    return dfs(len(nums) - 1)


from functools import cache


def rob_improved(nums: list[int]) -> int:

    @cache
    def helper(index):
        if index < 0:
            return 0
        take = nums[index] + helper(index - 2)
        skip = helper(index - 1)
        return max(take, skip)

    return helper(len(nums) - 1)
