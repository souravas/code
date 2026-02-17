def range_sum_query_immutable(nums: list[int], left: int, right: int) -> int:
    prefix_sum = [0]
    current_sum = 0

    for num in nums:
        current_sum += num
        prefix_sum.append(current_sum)

    return prefix_sum[right+1] - prefix_sum[left]


