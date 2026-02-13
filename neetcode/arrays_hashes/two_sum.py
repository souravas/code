class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        nums_map = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in nums_map.keys():
                return [nums_map[complement], i]
            nums_map[nums[i]] = i
        return []


if __name__ == "__main__":
    solution = Solution()
    nums = [1, 2, 3, 1, 4, 5]
    print(solution.twoSum(nums, 2))
