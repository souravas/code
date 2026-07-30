from math import inf
from functools import cache


def knapsack(weights: list[int], values: list[int], max_weight: int) -> int | float:

    @cache
    def solve(index, total_weight, total_value):
        if total_weight > max_weight:
            return -inf
        if index == len(weights):
            return total_value

        return max(
            solve(
                index + 1, total_weight + weights[index], total_value + values[index]
            ),
            solve(index + 1, total_weight, total_value),
        )

    return solve(0, 0, 0)


def knapsack_improved(
    weights: list[int], values: list[int], max_weight: int
) -> int | float:

    @cache
    def solve(index, capacity):
        if capacity <= 0 or index == len(weights):
            return 0

        take = 0
        if weights[index] <= capacity:
            take = values[index] + solve(index + 1, capacity - weights[index])
        skip = solve(index + 1, capacity)
        return max(take, skip)

    return solve(0, max_weight)
