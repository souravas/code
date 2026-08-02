def count_smaller(nums: list[int]) -> list[int]:
    if not nums:
        return []

    # Compress values to ranks 1..k so they can index a Fenwick tree.
    ranks = {v: i + 1 for i, v in enumerate(sorted(set(nums)))}
    size = len(ranks)
    tree = [0] * (size + 1)

    def update(i: int) -> None:  # record one occurrence of rank i
        while i <= size:
            tree[i] += 1
            i += i & -i

    def query(i: int) -> int:  # how many recorded ranks are <= i
        total = 0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    counts = [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        r = ranks[nums[i]]
        counts[i] = query(r - 1)  # r - 1 keeps it strictly smaller
        update(r)
    return counts
