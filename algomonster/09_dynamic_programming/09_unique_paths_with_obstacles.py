def unique_paths_ii(obstacle_grid: list[list[int]]) -> int:
    row_max = len(obstacle_grid) - 1
    col_max = len(obstacle_grid[0]) - 1

    def dfs(row, col):
        key = (row, col)
        if key in cache:
            return cache[key]
        if row == row_max and col == col_max:
            return 1
        if row < 0 or col < 0 or row > row_max or col > col_max:
            return 0
        if obstacle_grid[row][col] == 1:
            return 0

        cache[key] = dfs(row + 1, col) + dfs(row, col + 1)
        return cache[key]

    cache = {}
    return dfs(0, 0)
