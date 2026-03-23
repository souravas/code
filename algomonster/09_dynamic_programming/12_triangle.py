def minimum_total(triangle: list[list[int]]) -> int:

    def dfs(row, col):
        if row == len(triangle):
            return 0
        key = (row, col)
        if key in cache:
            return cache[key]

        cache[key] = triangle[row][col] + min(dfs(row + 1, col), dfs(row + 1, col + 1))
        return cache[key]

    cache = {}
    return dfs(0, 0)
