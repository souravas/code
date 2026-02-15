def subarray_sum_longest(nums: list[int], target: int) -> int:
    left = 0
    longest = 0
    current_sum = 0

    for right in range(len(nums)):
        current_sum += nums[right]

        while current_sum > target:
            current_sum -= nums[left]
            left += 1

        longest = max(longest, right - left + 1)

    return longest
