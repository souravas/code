from functools import cache


def bounded_knapsack(
    weights: list[int], values: list[int], quantities: list[int], capacity: int
) -> int:

    @cache
    def solve(index, remaining_capacity):
        if remaining_capacity <= 0 or index == len(weights):
            return 0

        max_value = 0
        max_k = min(quantities[index], remaining_capacity // weights[index])

        for k in range(max_k + 1):
            value = (values[index] * k) + solve(
                index + 1, remaining_capacity - (weights[index] * k)
            )
            max_value = max(value, max_value)

        return max_value

    return solve(0, capacity)
