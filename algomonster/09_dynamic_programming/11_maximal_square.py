def maximal_square(matrix: list[list[int]]) -> int:
    def dfs(row, col):
        if row < 0 or col < 0:
            return 0
        key = (row, col)
        if key in cache:
            return cache[key]

        if matrix[row][col] == 0:
            cache[key] = 0
        else:
            cache[key] = 1 + min(
                dfs(row - 1, col), dfs(row, col - 1), dfs(row - 1, col - 1)
            )
        return cache[key]

    cache = {}
    dfs(0, 0)
    result = 0

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result = max(result, dfs(i, j))

    return result * result
