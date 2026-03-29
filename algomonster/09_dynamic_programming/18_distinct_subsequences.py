from functools import cache


def distinct_subsequences(s: str, t: str) -> int:
    @cache
    def dfs(index1, index2):
        if index2 == len(t):
            return 1
        if index1 == len(s):
            return 0

        result = dfs(index1 + 1, index2)
        if s[index1] == t[index2]:
            result += dfs(index1 + 1, index2 + 1)

        return result

    return dfs(0, 0)
