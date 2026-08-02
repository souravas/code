def get_skyline(buildings: list[list[int]]) -> list[list[int]]:
    if not buildings:
        return []
    return _solve(buildings, 0, len(buildings) - 1)


def _solve(bs, lo, hi):
    if lo == hi:
        L, R, H = bs[lo]
        return [[L, H], [R, 0]]
    mid = (lo + hi) // 2
    return _merge(_solve(bs, lo, mid), _solve(bs, mid + 1, hi))


def _merge(A, B):
    out = []
    i = j = 0
    h1 = h2 = 0  # current height contributed by each side

    def push(x, h):
        if not out or out[-1][1] != h:
            out.append([x, h])

    while i < len(A) and j < len(B):
        if A[i][0] < B[j][0]:
            x, h1 = A[i]
            i += 1
        elif A[i][0] > B[j][0]:
            x, h2 = B[j]
            j += 1
        else:  # same x: consume both before emitting
            x, h1 = A[i]
            h2 = B[j][1]
            i += 1
            j += 1
        push(x, max(h1, h2))

    for x, h in A[i:]:  # the other side is exhausted, so its
        push(x, h)  # trailing height is 0 and max(0, h) == h
    for x, h in B[j:]:
        push(x, h)

    return out
