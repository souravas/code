class Solution:
    def isValid(self, s: str) -> bool:
        open_close_map = {'(': ')', '{': '}', '[': ']'}
        stack = []
        for char in s:
            if char in open_close_map.keys():
                stack.append(char)
            elif not stack or open_close_map[stack.pop()] != char:
                return False
        return not stack
