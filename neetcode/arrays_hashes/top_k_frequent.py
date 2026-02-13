from collections import Counter, defaultdict


class Solution:
    def topKFrequent1(self, nums: list[int], k: int) -> list[int]:
        counter = Counter(nums)
        return [element for element, count in counter.most_common(k)]

    def topKFrequent2(self, nums: list[int], k: int) -> list[int]:
        frequency = defaultdict(int)

        for num in nums:
            frequency[num] += 1

        return [num for num, freq in sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:k]]
