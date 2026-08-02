from collections import deque

directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
target = ((1, 2, 3), (4, 5, 0))


def num_steps(init_pos: list[list[int]]) -> int:
    init_pos_tuple = tuple(tuple(line) for line in init_pos)

    if init_pos_tuple == target:
        return 0
    queue = deque([init_pos_tuple])
    visited = set([init_pos_tuple])
    distance = 0

    while queue:
        n = len(queue)
        for _ in range(n):
            state = queue.popleft()
            if state == target:
                return distance
            row, col = 0, 0
            for i, line in enumerate(state):
                for j, entry in enumerate(line):
                    if entry == 0:
                        row, col = i, j
                        break

            for delta_row, delta_col in directions:
                new_row, new_col = row + delta_row, col + delta_col

                if 0 <= new_row < 2 and 0 <= new_col < 3:
                    new_state = [list(line) for line in state]
                    new_state[new_row][new_col], new_state[row][col] = (
                        new_state[row][col],
                        new_state[new_row][new_col],
                    )
                    new_tuples = tuple(tuple(line) for line in new_state)
                    if new_tuples not in visited:
                        visited.add(new_tuples)
                        queue.append(new_tuples)
        distance += 1
    return -1
