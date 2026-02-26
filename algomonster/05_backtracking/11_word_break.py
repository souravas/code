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
