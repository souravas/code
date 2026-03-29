from functools import cache


def shortest_common_supersequence(str1: str, str2: str) -> str:
    @cache
    def dfs(index1, index2):
        if index1 == len(str1) and index2 == len(str2):
            return ""
        if index1 == len(str1):
            return str2[index2:]
        if index2 == len(str2):
            return str1[index1:]

        if str1[index1] == str2[index2]:
            return str1[index1] + dfs(index1 + 1, index2 + 1)

        take_str1 = str1[index1] + dfs(index1 + 1, index2)
        take_str2 = str2[index2] + dfs(index1, index2 + 1)

        if len(take_str1) <= len(take_str2):
            return take_str1
        else:
            return take_str2

    return dfs(0, 0)
