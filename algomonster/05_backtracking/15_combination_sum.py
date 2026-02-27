def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    def dfs(index, nums, remaining):
        if remaining == 0:
            result.append(nums[:])
            return

        if remaining < 0:
            return

        for i in range(index, len(candidates)):
            nums.append(candidates[i])
            dfs(i, nums, remaining - candidates[i])
            nums.pop()

    candidates.sort()
    result = []
    dfs(0, [], target)
    return result
