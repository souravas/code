def largest_rectangle(heights: list[int]) -> int:
    stack = []  # indices, heights strictly increasing
    best = 0
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0  # sentinel flushes the stack
        while stack and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            left = stack[-1] + 1 if stack else 0  # first index still >= height
            best = max(best, height * (i - left))
        stack.append(i)
    return best
