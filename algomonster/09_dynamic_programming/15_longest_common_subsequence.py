from functools import cache


def longest_common_subsequence(word1: str, word2: str) -> int:
    @cache
    def dfs(i, j):
        if i == len(word1) or j == len(word2):
            return 0

        if word1[i] == word2[j]:
            return 1 + dfs(i + 1, j + 1)
        else:
            return max(dfs(i + 1, j), dfs(i, j + 1))

    return dfs(0, 0)
