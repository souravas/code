from functools import cache


def longest_string_chain(words: list[str]) -> int:
    words_map = {words[index]: index for index in range(len(words))}

    @cache
    def solve(index):
        current_word = words[index]
        longest = 0
        for i in range(len(current_word)):
            previous_word = current_word[:i] + current_word[i + 1 :]
            if previous_word in words_map:
                longest = max(longest, solve(words_map[previous_word]))
        return longest + 1

    longest = 0
    for i in range(len(words)):
        longest = max(longest, solve(i))
    return longest
