def container_with_most_water(arr: list[int]) -> int:
    left = 0
    right = len(arr) - 1
    max_water = 0

    while left < right:
        current = (right - left) * min(arr[left], arr[right])
        max_water = max(max_water, current)

        if arr[left] < arr[right]:
            left += 1
        else:
            right -= 1

    return max_water
