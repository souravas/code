# General Template (Top-Down)

# from functools import lru_cache


# def solve(items):
#     n = len(items)
#     FULL = (1 << n) - 1

#     @lru_cache(maxsize=None)
#     def dp(mask, extra):
#         if mask == FULL:  # all items selected
#             return base_case_value

#         result = initial_value  # inf / -inf / 0

#         for i in range(n):
#             if mask & (1 << i):  # already selected
#                 continue

#             new_mask = mask | (1 << i)
#             candidate = dp(new_mask, new_extra) + transition_cost
#             result = combine(result, candidate)

#         return result

#     return dp(initial_mask, initial_extra)


# Example: Minimum Cost Assignment

# dp(mask) = min(cost[worker][task] + dp(mask | (1 << task)))
#            over all unassigned task
# from functools import lru_cache

# def min_cost_assignment(cost):
#     n = len(cost)

#     @lru_cache(maxsize=None)
#     def dp(mask):
#         worker = mask.bit_count()  # number of assigned tasks
#         if worker == n:
#             return 0

#         result = float('inf')
#         for task in range(n):
#             if mask & (1 << task):
#                 continue
#             result = min(result,
#                          cost[worker][task] + dp(mask | (1 << task)))
#         return result

#     return dp(0)
