from collections import defaultdict


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        minimum_length = float("inf")
        substring_index = [-1, -1]
        t_map = defaultdict(int)
        s_map = defaultdict(int)

        for in_char in t:
            t_map[in_char] += 1

        need = len(t_map.keys())
        have = 0
        left = 0

        for right, in_char in enumerate(s):
            s_map[in_char] += 1
            if s_map[in_char] == t_map[in_char]:
                have += 1
            while need == have:
                current_length = right - left + 1
                if minimum_length > current_length:
                    minimum_length = current_length
                    substring_index = [left, right]
                out_char = s[left]
                s_map[out_char] -= 1
                if s_map[out_char] + 1 == t_map[out_char]:
                    have -= 1
                left += 1
        left, right = substring_index
        return s[left:right + 1]
