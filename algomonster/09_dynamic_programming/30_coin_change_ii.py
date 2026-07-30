from functools import cache


def coin_game(coins: list[int], amount: int) -> int:

    @cache
    def solve(index, current_amount):
        #  To avoid counting 1 + 2 and 2 + 1 as different answers,
        # each call only considers coins from the current index onward,
        # which fixes one order per combination.
        if current_amount == 0:
            return 1
        if current_amount < 0:
            return 0
        ways = 0

        for i in range(index, len(coins)):
            ways += solve(i, current_amount - coins[i])

        return ways

    return solve(0, amount)
