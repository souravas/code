from collections import defaultdict


class Solution:
    def isAnagram1(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)

    def isAnagram2(self, s: str, t: str) -> bool:
        alphabets = [0] * 26
        for character in s:
            alphabets[ord(character) - ord('a')] += 1
        for character in t:
            alphabets[ord(character) - ord('a')] -= 1
        for i in alphabets:
            if i != 0:
                return False
        return True

    def isAnagram3(self, s: str, t: str) -> bool:
        count = defaultdict(int)
        for character in s:
            count[character] += 1
        for character in t:
            count[character] -= 1
        for i in count.values():
            if i != 0:
                return False
        return True


if __name__ == "__main__":
    solution = Solution()
    print(solution.isAnagram1("abc", "cba"))
    print(solution.isAnagram2("abc", "bba"))
    print(solution.isAnagram3("abc", "bba"))
