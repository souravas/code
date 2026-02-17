def subarray_sum(arr: list[int], target: int) -> list[int]:
    prefix_sum = {0: 0}
    current_sum = 0

    for index in range(len(arr)):
        current_sum += arr[index]

        required = current_sum - target
        if required in prefix_sum:
            return [prefix_sum[required], index + 1]
        prefix_sum[current_sum] = index + 1

    return []


def subarray_sum_total(arr: list[int], target: int) -> int:
    prefix_sums = {0: 1}
    current_sum = 0
    result = 0

    for num in arr:
        current_sum += num
        required = current_sum - target
        if required in prefix_sums:
            result += prefix_sums[required]

        prefix_sums[current_sum] = prefix_sums.get(current_sum, 0) + 1
    return result
