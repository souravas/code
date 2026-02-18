def find_boundary(arr: list[bool]) -> int:
    left, right = 0, len(arr) - 1
    boundary_index = -1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid]:
            right = mid - 1
            boundary_index = mid
        else:
            left = mid + 1
    return boundary_index
