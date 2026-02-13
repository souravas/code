class Solution:
    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        stack = []
        result = [0] * len(temperatures)

        for i, t in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < t:
                index = stack.pop()
                result[index] = (i - index)
            stack.append(i)
        return result
