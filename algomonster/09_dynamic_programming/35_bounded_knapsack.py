# dp[i][j] = max(
#     dp[i - 1][j - x * w] + x * v
#     for all x where 0 <= x <= q and x * w <= j
# )


def bounded_knapsack(items, capacity):
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        weight, value, quantity = items[i - 1]
        for j in range(capacity + 1):
            best = dp[i - 1][j]  # take 0 copies
            for count in range(1, quantity + 1):
                total_weight = count * weight
                if total_weight > j:
                    break
                candidate = dp[i - 1][j - total_weight] + count * value
                best = max(best, candidate)
            dp[i][j] = best

    return dp[n][capacity]


def bounded_knapsack_with_binary_decomposition(items, capacity):
    expanded = []

    for weight, value, quantity in items:
        chunk = 1
        while chunk <= quantity:
            expanded.append((chunk * weight, chunk * value))
            quantity -= chunk
            chunk *= 2
        if quantity > 0:
            expanded.append((quantity * weight, quantity * value))

    dp = [0] * (capacity + 1)
    for w, v in expanded:
        for j in range(capacity, w - 1, -1):
            dp[j] = max(dp[j], dp[j - w] + v)

    return dp[capacity]
