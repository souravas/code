def merge_sort(arr):
    # base case
    if len(arr) <= 1:
        return arr

    # divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # conquer (merge two sorted halves)
    return merge(left, right)


def merge(left, right):
    i, j = 0, 0
    merged = []

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged
