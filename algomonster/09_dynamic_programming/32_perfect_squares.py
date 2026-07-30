from math import inf, sqrt
from functools import cache
import sys

sys.setrecursionlimit(10**6)


def perfect_squares(n: int) -> int | float:

    @cache
    def solve(number, current_sum):
        if current_sum == 0:
            return 0
        if current_sum < 0:
            return inf

        result = inf

        for i in range(number, int(sqrt(n)) + 1):
            result = min(result, 1 + solve(i, current_sum - (i * i)))

        return result

    result = solve(1, n)
    if result == inf:
        return -1
    return result


def perfect_squares_improved(n: int) -> int | float:

    @cache
    def solve(remaining):
        if remaining == 0:
            return 0
        if remaining < 0:
            return inf

        result = inf

        for i in range(1, int(sqrt(n)) + 1):
            result = min(result, 1 + solve(remaining - (i * i)))

        return result

    result = solve(n)
    if result == inf:
        return -1
    return result
