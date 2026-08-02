# def greedy(items):
#     # 1. Sort by the greedy key (deadline, end time, size, ratio, ...)
#     items.sort(key=greedy_key)
#     result = 0
#     state = initial_state()
#     for item in items:
#         # 2. Take the item only if the locally best choice is valid
#         if is_feasible(item, state):
#             result += take(item, state)
#     # 3. Correctness relies on an exchange argument, not just passing tests
#     return result


def make_change(coins: list[int], amount: int) -> int:
    # largest coin first
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        # take as many of this coin as fit
        count += amount // coin
        amount %= coin
    return count if amount == 0 else -1
