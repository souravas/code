def find_min_rotated(arr: list[int]) -> int:
    left = 0
    right = len(arr) - 1
    min_index = -1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] > arr[-1]:
            left = mid + 1
        else:
            right = mid - 1
            min_index = mid
    return min_index