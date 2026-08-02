from collections import defaultdict


class Solution:
    def isValidSudoku1(self, board: list[list[str]]) -> bool:
        def valid_rows():
            for row in board:
                seen = set()
                for element in row:
                    if element in seen and element != ".":
                        return False
                    seen.add(element)
            return True

        def valid_columns():
            for i in range(len(board)):
                seen = set()
                for j in range(len(board[0])):
                    if board[j][i] in seen and board[j][i] != ".":
                        return False
                    seen.add(board[j][i])
            return True

        def valid_grids():
            seen = defaultdict(set)
            for i in range(len(board)):
                for j in range(len(board[0])):
                    current = board[i][j]
                    if current in seen[(i // 3, j // 3)] and current != ".":
                        return False
                    seen[(i // 3, j // 3)].add(current)
            return True

        return valid_rows() and valid_columns() and valid_grids()

    def isValidSudoku2(self, board: list[list[str]]) -> bool:
        row = defaultdict(set)
        col = defaultdict(set)
        grid = defaultdict(set)

        for i in range(len(board)):
            for j in range(len(board[0])):
                current = board[i][j]
                if current == ".":
                    continue
                if (
                    current in row[i]
                    or current in col[j]
                    or current in grid[(i // 3, j // 3)]
                ):
                    return False
                row[i].add(current)
                col[j].add(current)
                grid[(i // 3), (j // 3)].add(current)
        return True
