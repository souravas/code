def two_sum_sorted(arr: list[int], target: int) -> list[int]:
    left = 0
    right = len(arr) - 1

    while left < right:
        current = arr[left] + arr[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1
    return []
