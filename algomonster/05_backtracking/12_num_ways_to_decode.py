def decode_ways(digits: str) -> int:
    def dfs(index):
        if index == len(digits):
            return 1
        if index in memo:
            return memo[index]

        if digits[index] == "0":
            return 0

        ways = 0
        ways += dfs(index + 1)

        # if digits[index] == "1":
        #     ways += dfs(index + 2)
        # if digits[index] == "2" and index < (len(digits) - 1):
        #     if int(digits[index + 1]) < 7:
        #         ways += dfs(index + 2)
        if 10 <= int(digits[index : (index + 2)]) <= 26:
            ways += dfs(index + 2)
        memo[index] = ways
        return memo[index]

    memo = {}
    return dfs(0)
