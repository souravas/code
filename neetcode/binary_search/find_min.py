class Solution:
    def findMin(self, nums):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                if mid > 0 and nums[mid] < nums[mid - 1]:
                    return nums[mid]
                right = mid - 1
        return nums[left]
