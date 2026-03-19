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
