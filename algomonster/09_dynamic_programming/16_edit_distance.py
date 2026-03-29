from functools import cache


def min_distance(word1: str, word2: str) -> int:
    @cache
    def solve(index1, index2):
        if index1 == len(word1) and index2 == len(word2):
            return 0
        if index1 == len(word1):
            return 1 + solve(index1, index2 + 1)
        if index2 == len(word2):
            return 1 + solve(index1 + 1, index2)

        if word1[index1] == word2[index2]:
            return solve(index1 + 1, index2 + 1)

        insert = 1 + solve(index1, index2 + 1)
        delete = 1 + solve(index1 + 1, index2)
        replace = 1 + solve(index1 + 1, index2 + 1)

        return min(insert, delete, replace)

    return solve(0, 0)
