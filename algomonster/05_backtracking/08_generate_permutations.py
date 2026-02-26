def permutations(letters: str) -> list[str]:
    def dfs(index):
        if index >= len(letters):
            result.append("".join(letters_array))
            return

        for i in range(index, len(letters)):
            letters_array[i], letters_array[index] = (
                letters_array[index],
                letters_array[i],
            )
            dfs(index + 1)
            letters_array[i], letters_array[index] = (
                letters_array[index],
                letters_array[i],
            )

    letters_array = list(letters)
    result = []
    dfs(0)
    return result
