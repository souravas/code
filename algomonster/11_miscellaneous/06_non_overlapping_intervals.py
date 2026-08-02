import heapq
from math import inf


def non_overlapping_intervals(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[1])

    kept = 0
    last_end = -inf
    for start, end in intervals:
        if start >= last_end:
            kept += 1
            last_end = end
        # else: overlaps, so we drop this one
    return len(intervals) - kept
