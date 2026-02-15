def remove_duplicates(arr: list[int]) -> int:
    slow = 0
    fast = 0

    while fast < len(arr):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
        fast += 1

    return slow + 1
