def basic_calculator(s: str) -> int:
    stack = []
    current_result = 0
    current_sign = 1
    index = 0

    while index < len(s):
        char = s[index]

        if char.isdigit():
            number = 0
            while index < len(s) and s[index].isdigit():
                number = number * 10 + int(s[index])
                index += 1
            current_result += current_sign * number
            continue
        if char == "+":
            current_sign = 1
        elif char == "-":
            current_sign = -1
        elif char == "(":
            stack.append(current_result)
            stack.append(current_sign)
            current_result = 0
            current_sign = 1
        elif char == ")":
            previous_sign = stack.pop()
            previous_result = stack.pop()
            current_result = previous_result + previous_sign * current_result

        index += 1

    return current_result
