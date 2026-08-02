from functools import cache


def coin_game(coins: list[int]) -> int:

    @cache
    def solve(i, j):
        if i == j:
            return coins[i]

        if i + 1 == j:
            return max(coins[i], coins[j])

        take_left = coins[i] + min(solve(i + 2, j), solve(i + 1, j - 1))

        take_right = coins[j] + min(solve(i + 1, j - 1), solve(i, j - 2))

        return max(take_left, take_right)

    return solve(0, len(coins) - 1)
