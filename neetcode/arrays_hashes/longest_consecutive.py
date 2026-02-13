class Solution:
    def longestConsecutive(self, nums: list[int]) -> int:
        seen = set(nums)
        result = 0
        for num in nums:
            if num - 1 not in seen and num in seen:
                count = 0
                while num in seen:
                    count += 1
                    num += 1
                result = max(result, count)
        return result
