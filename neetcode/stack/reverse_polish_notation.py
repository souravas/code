class Solution:
    def evalRPN(self, tokens: list[str]) -> int:
        operators = {"+", "-", "*", "/"}
        stack = []
        for token in tokens:
            if token in operators:
                right = stack.pop()
                left = stack.pop()
                stack.append(self.resolve(left, right, token))
            else:
                stack.append(int(token))
        return stack.pop()

    def resolve(self, left, right, token):
        if token == "+":
            return left + right
        elif token == "-":
            return left - right
        elif token == "*":
            return left * right
        else:
            return int(left / right)
