class SegmentTree:
    def __init__(self, arr):
        # 4n is a safe upper bound for the flat tree array
        self.tree = [0] * (4 * len(arr))
        for i in range(len(arr)):
            self.update(1, 0, len(arr) - 1, i, arr[i])

    # walk down to the leaf for idx, then recompute sums on the way up
    def update(self, cur, cur_left, cur_right, idx, val):
        if cur_left == cur_right:
            self.tree[cur] = val
            return
        cur_mid = (cur_left + cur_right) // 2
        if idx <= cur_mid:
            self.update(cur * 2, cur_left, cur_mid, idx, val)
        else:
            self.update(cur * 2 + 1, cur_mid + 1, cur_right, idx, val)
        self.tree[cur] = self.tree[cur * 2] + self.tree[cur * 2 + 1]

    # sum of arr[query_left..query_right]
    def query(self, cur, cur_left, cur_right, query_left, query_right):
        # current interval sits entirely outside the query
        if cur_left > query_right or cur_right < query_left:
            return 0
        # current interval sits entirely inside the query
        if query_left <= cur_left and cur_right <= query_right:
            return self.tree[cur]
        # partial overlap: recurse into both children
        cur_mid = (cur_left + cur_right) // 2
        return self.query(
            cur * 2, cur_left, cur_mid, query_left, query_right
        ) + self.query(cur * 2 + 1, cur_mid + 1, cur_right, query_left, query_right)
