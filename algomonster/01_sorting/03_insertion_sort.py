def insertion_sort(nums):
    for i in range(len(nums)):
        current = i
        while current > 0 and nums[current] < nums[current - 1]:
            nums[current], nums[current - 1] = nums[current - 1], nums[current]
            current -= 1
    return nums


nums = [4, 2, 1, 5, 6, 7, 3, 44, 33, 2, 1, 0]
print(insertion_sort(nums))
