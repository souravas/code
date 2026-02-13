from collections import defaultdict


class Solution:
    def groupAnagrams1(self, strs: list[str]) -> list[list[str]]:
        anagrams_map = {}
        for word in strs:
            sorted_word = ''.join(sorted(word))
            if sorted_word in anagrams_map:
                anagrams_map[sorted_word].append(word)
            else:
                anagrams_map[sorted_word] = [word]
        return list(anagrams_map.values())

    def groupAnagrams2(self, strs: list[str]) -> list[list[str]]:
        anagrams_map = defaultdict(list)
        for word in strs:
            sorted_word = ''.join(sorted(word))
            anagrams_map[sorted_word].append(word)
        return list(anagrams_map.values())


if __name__ == "__main__":
    solution = Solution()
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    print(solution.groupAnagrams1(strs))
    print(solution.groupAnagrams2(strs))
