def partition(s: str) -> list[list[str]]:
    result = []

    def dfs(index, current):
        if index >= len(s):
            result.append(current[:])
            return

        for i in range(index, len(s)):
            if is_palindrome(index, i):
                current.append(s[index : i + 1])
                dfs(i + 1, current)
                current.pop()

    def is_palindrome(i, j):
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True

    dfs(0, [])
    return result
