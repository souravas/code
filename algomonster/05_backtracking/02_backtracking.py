def letter_combination(n: int) -> list[str]:
    def dfs(index, current):
        if index == n:
            result.append("".join(current))
            return

        for letter in letters:
            current.append(letter)
            dfs(index + 1, current)
            current.pop()

    letters = ["a", "b"]
    result = []
    dfs(0, [])
    return result
