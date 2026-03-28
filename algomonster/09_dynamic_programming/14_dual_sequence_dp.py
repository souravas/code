def lcs(s1, s2):
    memo = {}

    def solve(i, j):
        if i == 0 or j == 0:
            return 0
        if (i, j) in memo:
            return memo[(i, j)]

        if s1[i - 1] == s2[j - 1]:
            result = 1 + solve(i - 1, j - 1)
        else:
            result = max(solve(i - 1, j), solve(i, j - 1))
        memo[(i, j)] = result
        return result

    return solve(len(s1), len(s2))


from functools import cache


def lcs_improved(s1, s2):

    @cache
    def solve(i, j):
        if i == 0 or j == 0:
            return 0

        if s1[i - 1] == s2[j - 1]:
            return 1 + solve(i - 1, j - 1)
        else:
            return max(solve(i - 1, j), solve(i, j - 1))

    return solve(len(s1), len(s2))
