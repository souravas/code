class SegmentTree:
    def __init__(self, arr: list[int]) -> None:
        self.tree = [0] * (4 * len(arr))
        for i in range(len(arr)):
            self.update(1, 0, len(arr) - 1, i, arr[i])

    def update(
        self,
        cur: int,
        cur_left: int,
        cur_right: int,
        idx: int,
        val: int,
    ) -> None:
        # make sure we reach leaf node when the left interval equals right interval and return the value located in the tree
        if cur_left == cur_right and cur_left == idx:
            self.tree[cur] = val
        else:
            # compute value of the midpoint where we cut the segment in half
            cur_mid = (cur_left + cur_right) // 2
            # remember n * 2 is left child node and n * 2 + 1 is the right child node
            if idx <= cur_mid:
                self.update(cur * 2, cur_left, cur_mid, idx, val)
            else:
                self.update(cur * 2 + 1, cur_mid + 1, cur_right, idx, val)
            # after updating the values, compute the new value for the node
            self.tree[cur] = max(self.tree[cur * 2], self.tree[cur * 2 + 1])

    def query(
        self,
        cur: int,
        cur_left: int,
        cur_right: int,
        query_left: int,
        query_right: int,
    ) -> int:
        # if our current left interval is greater than the queried right interval it means we are out of range
        # similarly, if the current right interval is less than the queried left interval we are out of range and in both cases return 0
        if cur_left > query_right or cur_right < query_left:
            return 0
        # check if we are in range, if we are return the current interval
        elif query_left <= cur_left and cur_right <= query_right:
            return self.tree[cur]
        # this means part of our interval is in range but part of our interval is not in range, we must therefore query both children
        cur_mid = (cur_left + cur_right) // 2
        return max(
            self.query(cur * 2, cur_left, cur_mid, query_left, query_right),
            self.query(cur * 2 + 1, cur_mid + 1, cur_right, query_left, query_right),
        )


def range_max(arr: list[int], operations: list[list[int]]) -> list[int]:
    tree = SegmentTree(arr)
    ans = []
    for op, a, b in operations:
        if op == 1:
            ans.append(tree.query(1, 0, len(arr) - 1, a, b))
        else:
            tree.update(1, 0, len(arr) - 1, a, b)
    return ans
