from math import inf


def dungeon_game(dungeon: list[list[int]]):
    rows = len(dungeon) - 1
    cols = len(dungeon[0]) - 1

    def dfs(row, col):
        key = (row, col)
        if key in cache:
            return cache[key]
        if row > rows or col > cols:
            return float("inf")

        if row == rows and col == cols:
            cache[key] = max(1, 1 - dungeon[row][col])
            return cache[key]

        next_need = min(dfs(row + 1, col), dfs(row, col + 1))
        need_here = next_need - dungeon[row][col]

        cache[key] = max(1, need_here)
        return cache[key]

    cache = {}
    return dfs(0, 0)
