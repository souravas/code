def min_cost_climbing_stairs(cost: list[int]) -> int:
    def dfs(index):
        if index >= len(cost):
            return 0
        if index in cache:
            return cache[index]
        cache[index] = min(dfs(index + 1), dfs(index + 2)) + cost[index]
        return cache[index]

    cache = {}
    return min(dfs(0), dfs(1))


from functools import cache


def min_cost_climbing_stairs_improved(cost: list[int]) -> int:

    @cache
    def helper(index):
        if index < 0:
            return 0
        return min(cost[index] + helper(index - 1), cost[index] + helper(index - 2))

    return min(helper(len(cost) - 1), helper(len(cost) - 2))
