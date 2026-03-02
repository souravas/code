INF = 2147483647

from collections import deque

INF = 2147483647

from collections import deque


def map_gate_distances(dungeon_map: list[list[int]]) -> list[list[int]]:
    queue = deque()
    visited = set()

    for i in range(len(dungeon_map)):
        for j in range(len(dungeon_map[0])):
            if dungeon_map[i][j] == 0:
                queue.append((i, j))

    level = 0
    neighbors = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    while queue:
        count = len(queue)
        for _ in range(count):
            row, col = queue.popleft()
            if (
                row < 0
                or col < 0
                or row == len(dungeon_map)
                or col == len(dungeon_map[0])
                or dungeon_map[row][col] == -1
            ):
                continue

            if dungeon_map[row][col] == INF:
                dungeon_map[row][col] = level
            for i, j in neighbors:
                new_row = row + i
                new_col = col + j
                if (new_row, new_col) in visited:
                    continue
                queue.append((new_row, new_col))
                visited.add((new_row, new_col))
        level += 1

    return dungeon_map


def map_gate_distances_optimized(dungeon_map: list[list[int]]) -> list[list[int]]:
    queue = deque()
    rows = len(dungeon_map)
    cols = len(dungeon_map[0])

    for i in range(len(dungeon_map)):
        for j in range(len(dungeon_map[0])):
            if dungeon_map[i][j] == 0:
                queue.append((i, j))

    neighbors = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    while queue:
        row, col = queue.popleft()
        for dr, dc in neighbors:
            new_row, new_col = row + dr, col + dc
            if (
                0 <= new_row < rows
                and 0 <= new_col < cols
                and dungeon_map[new_row][new_col] == INF
            ):
                dungeon_map[new_row][new_col] = dungeon_map[row][col] + 1
                queue.append((new_row, new_col))

    return dungeon_map
