def selection_sort(nums):
    for i in range(len(nums)):
        min_index = i
        for j in range(i, len(nums)):
            if nums[j] < nums[min_index]:
                min_index = j
        nums[min_index], nums[i] = nums[i], nums[min_index]
    return nums


nums = [4, 2, 1, 5, 6, 7, 3, 44, 33, 2, 1, 0]
print(selection_sort(nums))
