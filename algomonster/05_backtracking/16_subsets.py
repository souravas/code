def subsets(nums: list[int]) -> list[list[int]]:
    result = []
    nums.sort()

    def dfs(index, current):
        if index == len(nums):
            result.append(current[:])
            return

        current.append(nums[index])
        dfs(index + 1, current)
        current.pop()
        dfs(index + 1, current)

    dfs(0, [])
    return result
