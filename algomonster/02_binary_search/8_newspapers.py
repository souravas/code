def newspapers_split(newspapers_read_times: list[int], num_coworkers: int) -> int:
    def possible(mid):
        current = 0
        index = 0
        coworkers = num_coworkers

        while index < len(newspapers_read_times):
            if (newspapers_read_times[index] + current) <= mid:
                current += newspapers_read_times[index]
                index += 1
            else:
                coworkers -= 1
                if coworkers <= 0:
                    return False
                current = 0
        return True

    left = 1
    right = sum(newspapers_read_times)
    result = 0

    while left <= right:
        mid = (left + right) // 2

        if possible(mid):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result
