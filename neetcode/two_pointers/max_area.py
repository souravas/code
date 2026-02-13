class Solution:
    def maxArea(self, height: list[int]) -> int:
        left, right = 0, len(height) - 1
        result = 0
        while left < right:
            current = min(height[left], height[right]) * (right - left)
            result = max(result, current)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return result
