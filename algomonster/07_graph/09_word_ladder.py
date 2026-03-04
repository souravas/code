from collections import deque


def word_ladder(begin: str, end: str, word_list: list[str]) -> int:
    result = 0
    queue = deque([begin])
    visited = set([begin])

    def one_difference(start, end):
        difference = 0
        if len(start) != len(end):
            return False
        for i in range(len(start)):
            if start[i] != end[i]:
                difference += 1
        return difference == 1

    while queue:
        elements = len(queue)
        for _ in range(elements):
            previous_word = queue.popleft()
            if previous_word == end:
                return result
            for word in word_list:
                if word in visited:
                    continue
                if one_difference(word, previous_word):
                    queue.append(word)
                    visited.add(word)

        result += 1

    return -1


def word_ladder_optimized(begin: str, end: str, word_list: list[str]) -> int:
    result = 0
    queue = deque([begin])
    word_set = set(word_list)
    word_set.remove(begin)
    alphabets = "abcdefghijklmnopqrstuvwxyz"

    while queue:
        count = len(queue)
        for _ in range(count):
            current_word = queue.popleft()
            if current_word == end:
                return result
            for i in range(len(current_word)):
                for alphabet in alphabets:
                    new_word = current_word[:i] + alphabet + current_word[i + 1 :]
                    if new_word in word_set:
                        queue.append(new_word)
                        word_set.remove(new_word)

        result += 1

    return -1
