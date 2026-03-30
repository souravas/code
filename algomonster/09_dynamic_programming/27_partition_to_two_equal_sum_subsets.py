from functools import cache


def can_partition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2 == 1:
        return False
    required = total // 2

    @cache
    def solve(index, current_sum):
        if index == len(nums):
            return current_sum == required

        return solve(index + 1, current_sum) or solve(
            index + 1, current_sum + nums[index]
        )

    return solve(0, 0)
