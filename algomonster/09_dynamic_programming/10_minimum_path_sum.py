from math import inf


def min_path_sum(grid: list[list[int]]) -> int | float:
    row_max = len(grid) - 1
    col_max = len(grid[0]) - 1

    def dfs(row, col):
        key = (row, col)
        if key in cache:
            return cache[key]
        if row < 0 or col < 0 or row > row_max or col > col_max:
            return inf

        if row == row_max and col == col_max:
            cache[key] = grid[row][col]
            return cache[key]

        cache[key] = grid[row][col] + min(dfs(row + 1, col), dfs(row, col + 1))
        return cache[key]

    cache = {}
    return dfs(0, 0)
