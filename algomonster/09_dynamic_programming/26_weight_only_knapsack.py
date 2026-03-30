from functools import cache


def knapsack_weight_only(weights: list[int]) -> list[int]:
    result = []

    @cache
    def dfs(index, current):
        if index == len(weights):
            result.append(current)
            return
        dfs(index + 1, current)
        dfs(index + 1, current + weights[index])

    dfs(0, 0)
    return list(set(result))
