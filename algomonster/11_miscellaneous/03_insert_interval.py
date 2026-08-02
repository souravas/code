def insert_interval(
    intervals: list[list[int]], new_interval: list[int]
) -> list[list[int]]:
    intervals.append(new_interval)
    intervals.sort(key=lambda x: x[0])

    result = []

    for start, end in intervals:
        if result and result[-1][1] >= start:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])

    return result
