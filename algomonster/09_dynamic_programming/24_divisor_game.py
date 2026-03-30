from functools import cache


def divisor_game(n: int) -> bool:
    @cache
    def can_win(current):
        if current <= 1:
            return False
        for i in range(1, current // 2 + 1):
            if current % i == 0:
                if not can_win(current - i):
                    return True
        return False

    return can_win(n)
