def generate_parentheses(n: int) -> list[str]:
    def generate(index, open, close):
        if index == (2 * n):
            result.append("".join(current))
            return
        if open < n:
            current.append("(")
            generate(index + 1, open + 1, close)
            current.pop()
        if open > close:
            current.append(")")
            generate(index + 1, open, close + 1)
            current.pop()

    result = []
    current = []
    generate(0, 0, 0)
    return result
