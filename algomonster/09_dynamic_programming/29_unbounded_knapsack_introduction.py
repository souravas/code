def count_ways(coins, i, amount, memo):
    if amount == 0:
        return 1
    if i == 0 or amount < 0:
        return 0
    if (i, amount) in memo:
        return memo[(i, amount)]

    # Skip this denomination OR use it (stay on same i)
    result = count_ways(coins, i - 1, amount, memo) + count_ways(
        coins, i, amount - coins[i - 1], memo
    )
    memo[(i, amount)] = result
    return result
