def subarray_sum_fixed(nums: list[int], k: int) -> int:
    current_sum = 0
    for i in range(k):
        current_sum += nums[i]

    max_sum = current_sum

    left = 0
    for right in range(k, len(nums)):
        current_sum += nums[right]
        current_sum -= nums[left]
        left += 1
        max_sum = max(max_sum, current_sum)

    return max_sum
