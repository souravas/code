def merge_sort(nums):
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])

    return merge(left, right)


def merge(left, right):
    result = []
    left_pointer, right_pointer = 0, 0

    while left_pointer < len(left) and right_pointer < len(right):
        if left[left_pointer] <= right[right_pointer]:
            result.append(left[left_pointer])
            left_pointer += 1
        else:
            result.append(right[right_pointer])
            right_pointer += 1
    result.extend(left[left_pointer:])
    result.extend(right[right_pointer:])
    return result


nums = [4, 2, 1, 5, 6, 7, 3, 44, 33, 2, 1, 0]
print(merge_sort(nums))
