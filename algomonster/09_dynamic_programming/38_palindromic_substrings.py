from functools import cache


def palindromic_substrings(s: str) -> int:

    @cache
    def is_palindrome(i, j):
        if i >= j:
            return True
        if s[i] != s[j]:
            return False
        return is_palindrome(i + 1, j - 1)

    count = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            if is_palindrome(i, j):
                count += 1
    return count
