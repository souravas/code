def multiply_matrix(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    if not a or not a[0] or not b or not b[0]:
        return []

    n, m, p = len(a), len(a[0]), len(b[0])

    # Precompute nonzeros of B, grouped by row: b_nz[k] = [(col, val), ...]
    b_nz = [[(j, b[k][j]) for j in range(p) if b[k][j]] for k in range(m)]

    result = [[0] * p for _ in range(n)]
    for i in range(n):
        row_out = result[i]
        for k, a_ik in enumerate(a[i]):
            if a_ik:  # skip zeros in A
                for j, b_kj in b_nz[k]:  # skip zeros in B
                    row_out[j] += a_ik * b_kj
    return result
