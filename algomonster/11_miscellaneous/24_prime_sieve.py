from typing import List


def prime_sieve(n: int) -> List[bool]:
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    for i in range(2, n + 1, 1):
        if primes[i]:
            for j in range(i + i, n + 1, i):
                primes[j] = False
    return primes
