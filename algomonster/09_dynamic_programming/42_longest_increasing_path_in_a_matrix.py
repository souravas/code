from functools import cache


def longest_increasing_path_in_a_matrix(matrix: list[list[int]]) -> int:

    @cache
    def solve(row, col):
        length = 0
        for x, y in directions:
            current = matrix[row][col]
            new_row = row + x
            new_col = col + y

            # infinite loop is avoided because of the condition current >= matrix[new_row][new_col]
            if (
                new_row < 0
                or new_col < 0
                or new_row == len(matrix)
                or new_col == len(matrix[0])
                or current >= matrix[new_row][new_col]
            ):
                continue
            length = max(length, solve(new_row, new_col))
        return length + 1

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    longest = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            longest = max(longest, solve(i, j))
    return longest
