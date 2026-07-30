# dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight] + value)


# Space Optimization

# for item in items:
#     for j in range(capacity, weight - 1, -1):  # right to left
#         dp[j] = max(dp[j], dp[j - weight] + value)


# Optimization (max/min):

# dp[j] = max(dp[j], dp[j - weight] + value)
# Counting (number of ways):

# dp[j] += dp[j - weight]
# Feasibility (can we achieve target?):

# dp[j] = dp[j] or dp[j - weight]
