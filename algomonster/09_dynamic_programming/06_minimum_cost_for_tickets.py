def minimum_cost_for_tickets(days: list[int], costs: list[int]) -> int:
    def find_next_index(start, length):
        end = start + 1
        while end < len(days):
            if length >= (days[end] - days[start] + 1):
                end += 1
            else:
                break
        return end

    def dfs(index):
        if index >= len(days):
            return 0
        if index in memo:
            return memo[index]
        one_days = dfs(find_next_index(index, 1)) + costs[0]
        seven_days = dfs(find_next_index(index, 7)) + costs[1]
        thirty_days = dfs(find_next_index(index, 30)) + costs[2]

        memo[index] = min(one_days, seven_days, thirty_days)
        return memo[index]

    memo = {}
    return dfs(0)


from functools import cache


def minimum_cost_for_tickets_improved(days: list[int], costs: list[int]) -> int:
    days_set = set(days)

    @cache
    def dfs(day):
        if day < 0:
            return 0
        if day not in days_set:
            return dfs(day - 1)

        daily = costs[0] + dfs(day - 1)
        weekly = costs[1] + dfs(day - 7)
        monthly = costs[2] + dfs(day - 30)

        return min(daily, weekly, monthly)

    return dfs(days[-1])
