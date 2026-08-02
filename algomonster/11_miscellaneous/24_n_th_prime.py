def nth_prime(n: int) -> int:
    primes = [True] * (100001)
    primes[0] = False
    primes[1] = False
    ans = 0
    cnt = 0
    for i in range(2, 100001):
        if primes[i]:
            cnt += 1
            if cnt == n:
                ans = i
                break
            for j in range(i + i, 100001, i):
                primes[j] = False
    return ans
