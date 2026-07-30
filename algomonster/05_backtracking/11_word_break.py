def word_break(s: str, words: list[str]) -> bool:

    def dfs(index):
        if index == len(s):
            return True
        if index in memo:
            return memo[index]

        for word in words:
            current = s[index : index + len(word)]
            if current != word:
                continue
            if dfs(index + len(word)):
                memo[index] = True
                return memo[index]
        memo[index] = False
        return memo[index]

    memo = {}
    return dfs(0)


from functools import cache


def word_break_cache(s: str, words: list[str]) -> bool:

    @cache
    def dfs(index):
        if index == len(s):
            return True

        for word in words:
            required_word = s[index : index + len(word)]
            if required_word != word:
                continue
            if dfs(index + len(word)):
                return True
        return False

    return dfs(0)
