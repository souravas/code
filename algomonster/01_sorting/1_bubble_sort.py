def bubble_sort(nums):
    for i in range(len(nums)):
        for j in range(0, len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums


nums = [4, 2, 1, 5, 6, 7, 3, 44, 33, 2, 1, 0]
print(bubble_sort(nums))
