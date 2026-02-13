class Solution:
    def trap(self, height: list[int]) -> int:
        size = len(height)
        prefix = [0] * size
        suffix = [0] * size
        prefix[0] = height[0]
        suffix[-1] = height[-1]
        result = 0
        for i in range(1, size):
            prefix[i] = max(prefix[i - 1], height[i])
        for i in range(size - 2, -1, -1):
            suffix[i] = max(suffix[i + 1], height[i])
        for i in range(size):
            result += min(prefix[i], suffix[i]) - height[i]
        return result
