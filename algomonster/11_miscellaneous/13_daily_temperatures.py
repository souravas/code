def daily_temperatures(t: list[int]) -> list[int]:
    result = [0] * len(t)
    stack = []

    for index, temperature in enumerate(t):
        while stack and t[stack[-1]] < temperature:
            j = stack.pop()
            result[j] = index - j
        stack.append(index)

    return result
