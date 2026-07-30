from math import inf
from functools import cache


def coin_change(coins: list[int], amount: int) -> int | float:

    @cache
    def solve(index, current_amount, count):
        if current_amount == 0:
            return count
        if current_amount < 0:
            return inf

        result = inf
        for i in range(index, len(coins)):
            result = min(result, solve(i, current_amount - coins[i], count + 1))

        return result

    result = solve(0, amount, 0)
    if result == inf:
        return -1
    return result


def coin_change_improved(coins: list[int], amount: int) -> int | float:

    @cache
    def solve(index, current_amount):
        if current_amount == 0:
            return 0
        if current_amount < 0:
            return inf

        result = inf
        for i in range(index, len(coins)):
            result = min(result, 1 + solve(i, current_amount - coins[i]))

        return result

    result = solve(0, amount)
    if result == inf:
        return -1
    return result


def coin_change_improved_further(coins: list[int], amount: int) -> int | float:

    @cache
    def solve(current_amount):
        if current_amount == 0:
            return 0
        if current_amount < 0:
            return inf

        result = inf
        for i in range(len(coins)):
            result = min(result, 1 + solve(current_amount - coins[i]))

        return result

    result = solve(amount)
    if result == inf:
        return -1
    return result
