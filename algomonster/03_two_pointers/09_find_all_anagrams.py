def find_all_anagrams(original: str, check: str) -> list[int]:
    result = []

    if len(original) < len(check):
        return result

    check_count = [0] * 26
    original_count = [0] * 26

    for i in range(len(check)):
        check_count[ord(check[i]) - ord("a")] += 1
        original_count[ord(original[i]) - ord("a")] += 1

    if check_count == original_count:
        result.append(0)

    for i in range(len(check), len(original)):
        original_count[ord(original[i]) - ord("a")] += 1
        original_count[ord(original[i - len(check)]) - ord("a")] -= 1
        if original_count == check_count:
            result.append(i - len(check) + 1)
    return result
