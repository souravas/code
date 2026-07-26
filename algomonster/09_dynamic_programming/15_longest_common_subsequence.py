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


from functools import cache


def longest_common_subsequence_improved(word1: str, word2: str) -> int:

    @cache
    def dfs(index1, index2):
        if index1 < 0 or index2 < 0:
            return 0
        if word1[index1] != word2[index2]:
            return max(dfs(index1 - 1, index2), dfs(index1, index2 - 1))
        return 1 + (dfs(index1 - 1, index2 - 1))

    return dfs(len(word1) - 1, len(word2) - 1)
