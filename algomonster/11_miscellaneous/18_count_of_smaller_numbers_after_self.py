def count_smaller(nums: list[int]) -> list[int]:
    counts = [0] * len(nums)

    def sort(pairs):
        if len(pairs) <= 1:
            return pairs
        mid = len(pairs) // 2
        left, right = sort(pairs[:mid]), sort(pairs[mid:])

        merged = []
        l = r = 0
        while l < len(left) and r < len(right):
            if left[l][1] <= right[r][1]:  # ties go left: equal is not smaller
                counts[left[l][0]] += r
                merged.append(left[l])
                l += 1
            else:
                merged.append(right[r])
                r += 1
        for i in range(l, len(left)):  # left leftovers: all of right was smaller
            counts[left[i][0]] += r
        merged.extend(left[l:])
        merged.extend(right[r:])
        return merged

    sort(list(enumerate(nums)))
    return counts
