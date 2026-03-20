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
