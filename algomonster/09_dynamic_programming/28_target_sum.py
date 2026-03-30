from functools import cache


def target_sum(nums: list[int], target: int) -> int:
    @cache
    def solve(index, current):
        if index == len(nums):
            if current == target:
                return 1
            else:
                return 0

        result = solve(index + 1, current + nums[index]) + solve(
            index + 1, current - nums[index]
        )
        return result

    return solve(0, 0)
