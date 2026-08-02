def interval_pattern(intervals):
    # 1. Sort by start time
    intervals.sort(key=lambda x: x[0])
    result = []
    for start, end in intervals:
        # 2. Overlap with the last kept interval? end >= next start
        if result and start <= result[-1][1]:
            # Merge: extend the end of the last interval
            result[-1][1] = max(result[-1][1], end)
        else:
            # 3. No overlap: keep this interval as-is
            result.append([start, end])
    return result
