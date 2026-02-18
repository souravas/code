def find_first_occurrence(arr: list[int], target: int) -> int:
    left, right = 0, len(arr) - 1
    first_index = -1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            first_index = mid
            right = mid - 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return first_index
