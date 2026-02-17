from collections import defaultdict, Counter


def get_minimum_window(original: str, check: str) -> str:
    def update_window(window, left, right):
        if not window:
            return [left, right]

        old_length = window[1] - window[0] + 1
        new_length = right - left + 1

        if new_length < old_length:
            return [left, right]

        if new_length == old_length:
            old_string = original[window[0] : window[1] + 1]
            new_string = original[left : right + 1]
            if new_string < old_string:
                return [left, right]
        return window

    check_map = Counter(check)
    required = len(check_map.keys())
    satisfied = 0

    original_map = defaultdict(int)
    left = 0
    window = []
    for right in range(len(original)):
        current = original[right]
        if current not in check_map:
            continue
        original_map[current] += 1
        if original_map[current] != check_map[current]:
            continue
        satisfied += 1
        while satisfied == required:
            window = update_window(window, left, right)
            remove_value = original[left]
            left += 1
            if remove_value not in check_map:
                continue
            original_map[remove_value] -= 1
            if original_map[remove_value] < check_map[remove_value]:
                satisfied -= 1

    if not window:
        return ""
    return original[window[0] : window[1] + 1]
