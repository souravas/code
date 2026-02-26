from math import inf


def coin_change(coins: list[int], amount: int):
    def dfs(amount):
        if amount == 0:
            return 0

        if amount < 0:
            return inf
        if amount in memo:
            return memo[amount]

        result = inf

        for coin in coins:
            current_result = dfs(amount - coin)
            if current_result == inf:
                continue
            result = min(result, current_result + 1)

        memo[amount] = result
        return memo[amount]

    memo = {}
    result = dfs(amount)
    if result == inf:
        return -1
    return result
