from functools import cache


def partition_array_for_maximum_sum(arr: list[int], k: int) -> int:
    n = len(arr)

    @cache
    def dfs(i):
        if i == n:
            return 0

        max_sum = 0
        current_max = 0
        length = 0

        for j in range(i, min(i + k, n)):
            length += 1
            current_max = max(current_max, arr[j])
            current_partition_sum = current_max * length
            max_sum = max(max_sum, current_partition_sum + dfs(j + 1))
        return max_sum

    return dfs(0)


from functools import cache


def partition_array_for_maximum_sum_alternative(arr: list[int], k: int) -> int:

    @cache
    def dfs(index):
        if index >= len(arr):
            return 0

        max_sum = 0
        for i in range(index + 1, min(index + k + 1, len(arr) + 1)):
            current_max = 0
            for j in range(index, i):
                current_max = max(current_max, arr[j])
            current_result = (current_max * (i - index)) + dfs(i)
            max_sum = max(max_sum, current_result)
        return max_sum

    return dfs(0)
