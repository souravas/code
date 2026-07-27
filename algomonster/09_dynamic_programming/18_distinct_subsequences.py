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


from functools import cache


def distinct_subsequences_improved(s: str, t: str) -> int:
    @cache
    def dfs(index1, index2):
        if index2 < 0:
            return 1
        if index1 < 0:
            return 0
        result = 0
        if s[index1] == t[index2]:
            result += dfs(index1 - 1, index2 - 1)
        result += dfs(index1 - 1, index2)
        return result

    return dfs(len(s) - 1, len(t) - 1)
