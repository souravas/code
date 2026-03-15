def fib_top_down(n, memo):
    if n in memo:
        return memo[n]

    if n == 0 or n == 1:
        return n

    result = fib_top_down(n - 1, memo) + fib_top_down(n - 2, memo)
    memo[n] = result
    return result


def fib_bottom_up(n):
    dp = [0, 1]
    for i in range(2, n + 1):
        dp.append(dp[i - 1] + dp[i - 2])
    return dp[-1]
