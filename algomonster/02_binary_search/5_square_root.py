def square_root(n: int) -> int:
    left = 1
    right = n
    result = 0

    while left <= right:
        mid = (left + right) // 2
        current = mid * mid
        if current <= n:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result