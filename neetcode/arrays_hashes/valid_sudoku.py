from collections import defaultdict


class Solution:
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        row = defaultdict(set)
        col = defaultdict(set)
        grid = defaultdict(set)

        for i in range(len(board)):
            for j in range(len(board[0])):
                current = board[i][j]
                if current == '.':
                    continue
                if current in row[i] or current in col[j] or current in grid[(i // 3, j // 3)]:
                    return False
                row[i].add(current)
                col[j].add(current)
                grid[(i // 3), (j // 3)].add(current)
        return True
