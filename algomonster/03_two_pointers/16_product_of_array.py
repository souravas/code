def product_of_array_except_self(nums: list[int]) -> list[int]:
    length = len(nums)
    left_prefix = [1]
    right_prefix = [1]

    for i in range(length):
        left_prefix.append(nums[i] * left_prefix[-1])
    for i in range(length):
        right_prefix.append(nums[length - i - 1] * right_prefix[-1])

    right_prefix.reverse()
    print(left_prefix)
    print(right_prefix)

    result = []

    for i in range(length):
        current = left_prefix[i] * right_prefix[i + 1]
        result.append(current)
    return result


nums = [1, 2, 3, 4]

product_of_array_except_self(nums)
# [1  1  2  6 _]
# [_ 24 12  4 1]
# Output: [24, 12, 8, 6]
