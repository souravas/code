from functools import lru_cache


def count_palindromes(s):
    n = len(s)

    @lru_cache(maxsize=None)
    def is_palin(l, r):
        if l >= r:
            return True
        return s[l] == s[r] and is_palin(l + 1, r - 1)

    return sum(is_palin(l, r) for l in range(n) for r in range(l, n))


def count_palindromes_bottom_up(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]

    # Base case: single characters (length 1)
    count = 0
    for i in range(n):
        dp[i][i] = True
        count += 1

    for length in range(2, n + 1):  # substring length: 2, 3, ..., n
        for l in range(n - length + 1):  # l + length - 1 must be < n
            r = l + length - 1  # right endpoint
            dp[l][r] = s[l] == s[r] and (length == 2 or dp[l + 1][r - 1])
            if dp[l][r]:
                count += 1

    return count
