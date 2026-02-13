class Solution:
    def generateParenthesis(self, n: int) -> list[str]:
        def generate(current, open, close):
            if len(current) == n * 2:
                result.append(current)
                return

            if open < n:
                generate(current + "(", open + 1, close)
            if close < open:
                generate(current + ')', open, close + 1)

        result = []
        generate("", 0, 0)
        return result
