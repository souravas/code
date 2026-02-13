class Solution:
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        row, col = len(matrix), len(matrix[0])

        top, bottom = 0, row - 1
        while top <= bottom:
            mid = (top + bottom) // 2
            if target > matrix[mid][-1]:
                top = mid + 1
            elif target < matrix[mid][0]:
                bottom = mid - 1
            else:
                break

        if top > bottom:
            return False

        target_row = (top + bottom) // 2
        left, right = 0, col - 1

        while left <= right:
            mid = (left + right) // 2
            if target > matrix[target_row][mid]:
                left = mid + 1
            elif target < matrix[target_row][mid]:
                right = mid - 1
            else:
                return True
        return False
