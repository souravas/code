def climbing_stairs(n: int) -> int:
    def dfs(n):
        if n in memo:
            return memo[n]
        if n == 1 or n == 2:
            return n
        memo[n] = dfs(n - 1) + dfs(n - 2)
        return memo[n]

    memo = {}
    return dfs(n)
