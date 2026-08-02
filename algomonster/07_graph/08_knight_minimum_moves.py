from collections import deque


def get_knight_shortest_path(x: int, y: int) -> int:
    queue = deque([(0, 0)])
    steps = 0
    visited = set()
    movements = [(-1, -2), (-2, -1), (-2, 1), (-1, 2), (1, -2), (2, -1), (2, 1), (1, 2)]

    while queue:
        elements = len(queue)
        for _ in range(elements):
            row, col = queue.popleft()
            if row == x and col == y:
                return steps
            for movement_x, movement_y in movements:
                new_row = row + movement_x
                new_col = col + movement_y
                if (new_row, new_col) in visited:
                    continue
                queue.append((new_row, new_col))
        steps += 1

    return -1
