class Solution:
    def containsDuplicate1(self, nums: list[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

    def containsDuplicate2(self, nums: list[int]) -> bool:
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                return True
        return False

    def containsDuplicate3(self, nums: list[int]) -> bool:
        seen = {}
        for num in nums:
            if num in seen:
                return True
            seen[num] = True
        return False


if __name__ == "__main__":
    solution = Solution()
    nums = [1, 2, 3, 1, 4, 5]
    print(solution.containsDuplicate1(nums))
    print(solution.containsDuplicate2(nums))
    print(solution.containsDuplicate3(nums))
