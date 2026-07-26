def unique_paths(m: int, n: int) -> int:
    m -= 1
    n -= 1

    def dfs(row, col):
        if (row, col) in cache:
            return cache[(row, col)]
        if row == m and col == n:
            return 1
        if row > m or col > n or row < 0 or col < 0:
            return 0
        cache[(row, col)] = dfs(row + 1, col) + dfs(row, col + 1)
        return cache[(row, col)]

    cache = {}
    return dfs(0, 0)


from functools import cache


def unique_paths_improved(m: int, n: int) -> int:

    @cache
    def dfs(m, n):
        if m == 0 and n == 0:
            return 1
        if m < 0 or n < 0:
            return 0
        return dfs(m - 1, n) + dfs(m, n - 1)

    return dfs(m - 1, n - 1)
