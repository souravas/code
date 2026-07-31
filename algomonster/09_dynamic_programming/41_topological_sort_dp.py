# def longestPath(matrix):
#     memo = {}

#     def dfs(r, c):
#         if (r, c) in memo:
#             return memo[(r, c)]

#         best = 1  # At minimum, the path includes this cell
#         for nr, nc in neighbors(r, c):
#             if matrix[nr][nc] > matrix[r][c]:
#                 best = max(best, 1 + dfs(nr, nc))

#         memo[(r, c)] = best
#         return best

#     return max(dfs(r, c) for all cells)
