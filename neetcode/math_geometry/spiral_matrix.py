class Solution:
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        result = []
        left, right = 0, len(matrix[0]) - 1
        top, bottom = 0, len(matrix) - 1

        while left <= right and top <= bottom:
            for i in range(left, right + 1):
                result.append(matrix[top][i])
            top += 1

            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1

            # a single remaining row or column is fully consumed by the two
            # loops above — without this the bottom/left passes re-read it
            if not (left <= right and top <= bottom):
                break

            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1

            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

        return result
