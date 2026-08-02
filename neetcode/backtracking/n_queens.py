from typing import List


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def is_valid(board, row, col):
            for index in range(n):
                if board[row][index] == "Q":
                    if index != col:
                        return False
            for index in range(n):
                if board[index][col] == "Q":
                    if index != row:
                        return False

            temp_row = row - 1
            temp_col = col - 1
            while temp_row >= 0 and temp_col >= 0:
                if board[temp_row][temp_col] == "Q":
                    return False
                temp_row -= 1
                temp_col -= 1

            temp_row = row - 1
            temp_col = col + 1
            while temp_row >= 0 and temp_col < n:
                if board[temp_row][temp_col] == "Q":
                    return False
                temp_row -= 1
                temp_col += 1
            return True

        def create_board(board):
            new_board = []
            for row in board:
                new_board.append("".join(row))
            return new_board

        def backtrack(current, row):
            if row == n:
                result.append(create_board(current))
                return
            for j in range(n):
                current[row][j] = "Q"
                if is_valid(current, row, j):
                    backtrack(current, row + 1)
                current[row][j] = "."

        result = []
        current = [["."] * n for i in range(n)]
        backtrack(current, 0)
        return result


if __name__ == "__main__":
    solution = Solution()
    print(solution.solveNQueens(4))
