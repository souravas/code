class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        n = len(matrix)

        # transpose reflects across the main diagonal, which is a rotation
        # composed with a horizontal flip — reversing each row undoes the flip
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        for i in range(n):
            for j in range(n // 2):
                matrix[i][j], matrix[i][n - j - 1] = matrix[i][n - j - 1], matrix[i][j]
