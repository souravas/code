def letter_combinations_of_phone_number(digits: str) -> list[str]:
    letters_map = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }
    result = []

    def dfs(index, current):
        if index == len(digits):
            result.append("".join(current))
            return

        letters = letters_map[digits[index]]
        for letter in letters:
            current.append(letter)
            dfs(index + 1, current)
            current.pop()

    dfs(0, [])
    return result
