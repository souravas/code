class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False

        s1_count = [0] * 26
        s2_count = [0] * 26

        for i in range(len(s1)):
            s1_count[ord(s1[i]) - ord('a')] += 1
            s2_count[ord(s2[i]) - ord('a')] += 1

        matches = 0
        for i in range(len(s1_count)):
            matches += 1 if s1_count[i] == s2_count[i] else 0

        left = 0
        for right in range(len(s1), len(s2)):
            if matches == 26:
                return True
            in_index = ord(s2[right]) - ord('a')
            s2_count[in_index] += 1
            if s2_count[in_index] == s1_count[in_index]:
                matches += 1
            elif s2_count[in_index] - 1 == s1_count[in_index]:
                matches -= 1

            out_index = ord(s2[left]) - ord('a')
            s2_count[out_index] -= 1
            if s2_count[out_index] == s1_count[out_index]:
                matches += 1
            elif s2_count[out_index] + 1 == s1_count[out_index]:
                matches -= 1
            left += 1
        return matches == 26
