from functools import cache


def longest_palindromic_subsequence(s: str) -> int:

    @cache
    def solve(i, j):
        if i == j:
            return 1
        if i > j:
            return 0

        if s[i] == s[j]:
            return 2 + solve(i + 1, j - 1)

        return max(solve(i + 1, j), solve(i, j - 1))

    return solve(0, len(s) - 1)
