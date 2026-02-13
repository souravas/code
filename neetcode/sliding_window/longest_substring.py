class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        seen = set()
        left = right = 0
        result = 0
        while right < len(s):
            current = s[right]
            while current in seen:
                seen.remove(s[left])
                left += 1
            else:
                right += 1
            seen.add(current)
            result = max(result, len(seen))

        return result
