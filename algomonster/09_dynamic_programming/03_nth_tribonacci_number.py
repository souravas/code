def nth_tribonacci_number(n: int) -> int:
    def dfs(n):
        if n in memo:
            return memo[n]
        memo[n] = dfs(n - 1) + dfs(n - 2) + dfs(n - 3)
        return memo[n]

    memo = {0: 0, 1: 1, 2: 1}
    return dfs(n)
