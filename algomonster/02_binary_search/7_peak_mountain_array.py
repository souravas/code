def peak_of_mountain_array(arr: list[int]) -> int:
    left = 0
    right = len(arr) - 1
    peak = -1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] > arr[mid + 1]:
            peak = mid
            right = mid - 1
        else:
            left = mid + 1

    return peak


arr = [0, 10, 3, 2, 1, 0]
print(peak_of_mountain_array(arr))
