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
