from typing import List


class Solution:
    def subsets1(self, nums: list[int]) -> list[list[int]]:
        result = []

        def subset_helper(current_nums, index):
            if index == len(nums):
                result.append(current_nums.copy())
                return
            current_nums.append(nums[index])
            subset_helper(current_nums, index + 1)
            current_nums.pop()
            subset_helper(current_nums, index + 1)

        subset_helper([], 0)
        return result

    def subsets2(self, nums: List[int]) -> List[List[int]]:
        def helper(index, current):
            result.append(current[:])

            for i in range(index, len(nums)):
                current.append(nums[i])
                helper(i + 1, current)
                current.pop()

        result = []
        helper(0, [])
        return result


# https://leetcode.com/problems/subsets/solutions/27281/a-general-approach-to-backtracking-questions-in-java-subsets-permutations-combination-sum-palindrome-partitioning/
