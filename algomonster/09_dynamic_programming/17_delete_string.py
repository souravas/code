from functools import cache


def delete_string(costs: list[int], s1: str, s2: str) -> int:
    @cache
    def dfs(index1, index2):
        if index1 == len(s1) and index2 == len(s2):
            return 0
        if index1 == len(s1):
            return costs[ord(s2[index2]) - ord("a")] + dfs(index1, index2 + 1)
        if index2 == len(s2):
            return costs[ord(s1[index1]) - ord("a")] + dfs(index1 + 1, index2)

        if s1[index1] == s2[index2]:
            return dfs(index1 + 1, index2 + 1)

        delete_s1 = costs[ord(s1[index1]) - ord("a")] + dfs(index1 + 1, index2)
        delete_s2 = costs[ord(s2[index2]) - ord("a")] + dfs(index1, index2 + 1)

        return min(delete_s1, delete_s2)

    return dfs(0, 0)
