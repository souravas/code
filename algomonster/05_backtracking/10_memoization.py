def fib(n):
    if n == 0 or n == 1:
        return n

    return fib(n - 1) + fib(n - 2)


def fib_improved(n, memo):
    if n in memo:
        return memo[n]

    if n == 0 or n == 1:
        return n

    result = fib_improved(n - 1, memo) + fib_improved(n - 2, memo)

    memo[n] = result

    return result
