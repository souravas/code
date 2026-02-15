def is_palindrome(s: str) -> bool:
    s = s.lower()
    left = 0
    right = len(s) - 1

    while left < right:
        if not ("a" <= s[left] <= "z"):
            left += 1
        elif not ("a" <= s[right] <= "z"):
            right -= 1
        elif s[left] != s[right]:
            return False
        else:
            left += 1
            right -= 1
    return True
