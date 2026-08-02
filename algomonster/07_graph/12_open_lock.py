from collections import deque


def num_steps(target_combo: str, trapped_combos: list[str]) -> int:
    next_steps = {str(x): str(x + 1) for x in range(10)}
    next_steps["9"] = "0"
    prev_steps = {value: key for key, value in next_steps.items()}

    queue = deque(["0000"])
    visited = set(["0000"])
    result = 0
    trapped_combos_set = set(trapped_combos)

    while queue:
        count = len(queue)
        for _ in range(count):
            current_num = queue.popleft()
            if current_num == target_combo:
                return result
            for i in range(4):
                new_num1 = (
                    current_num[:i] + next_steps[current_num[i]] + current_num[i + 1 :]
                )
                if new_num1 not in trapped_combos_set and new_num1 not in visited:
                    visited.add(new_num1)
                    queue.append(new_num1)

                new_num2 = (
                    current_num[:i] + prev_steps[current_num[i]] + current_num[i + 1 :]
                )
                if new_num2 not in trapped_combos_set and new_num2 not in visited:
                    visited.add(new_num2)
                    queue.append(new_num2)
        result += 1

    return -1
