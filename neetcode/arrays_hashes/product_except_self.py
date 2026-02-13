class Solution:
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        n = len(nums)
        left, right = [1] * n, [1] * n
        for i in range(1, n):
            left[i] = left[i - 1] * nums[i - 1]
            right[-i - 1] = right[-i] * nums[-i]
        return [l * r for l, r in zip(left, right)]
