from collections import defaultdict

def get_minimum_window(original: str, check: str) -> str:
    check_map = defaultdict(int)
    original_map = defaultdict(int)
    minimum_length = len(original) + 1
    minimum_index = -1
    satisfied = 0
    left_index = 0

    for character in check:
        check_map[character] += 1
    required = len(check_map.keys())

    for current_index in range(len(original)):
        current_character = original[current_index]
        original_map[current_character] += 1
        if current_character in check_map:
            if original_map[current_character] == check_map[current_character]:
                satisfied += 1

        while satisfied == required:
            current_length = current_index - left_index + 1
            if current_length < minimum_length:
                minimum_length = current_length
                minimum_index = left_index
            elif current_length == minimum_length:
                previous_word = original[minimum_index:minimum_index+minimum_length]
                current_word = original[left_index:current_index+1]
                if current_word < previous_word:
                    minimum_index = left_index
            original_map[original[left_index]] -= 1
            if original[left_index] in check_map:
                if check_map[original[left_index]] > original_map[original[left_index]]:
                    satisfied -= 1
            left_index += 1

    if minimum_index == -1:
        return ""
    return original[minimum_index:minimum_index+minimum_length]
