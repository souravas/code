def flood_fill_dfs(
    r: int, c: int, replacement: int, image: list[list[int]]
) -> list[list[int]]:
    num_rows = len(image)
    num_cols = len(image[0])

    original_color = image[r][c]
    visited = set()

    def dfs(row, col):
        if (
            row < 0
            or col < 0
            or row == num_rows
            or col == num_cols
            or image[row][col] != original_color
            or (row, col) in visited
        ):
            return

        image[row][col] = replacement
        visited.add((row, col))

        dfs(row + 1, col)
        dfs(row - 1, col)
        dfs(row, col + 1)
        dfs(row, col - 1)

    dfs(r, c)
    return image


from collections import deque


def flood_fill_bfs(
    r: int, c: int, replacement: int, image: list[list[int]]
) -> list[list[int]]:
    num_rows = len(image)
    num_cols = len(image[0])

    original_color = image[r][c]
    visited = set()
    queue = deque([(r, c)])

    while queue:
        row, col = queue.popleft()
        if (
            row < 0
            or col < 0
            or row == num_rows
            or col == num_cols
            or image[row][col] != original_color
            or (row, col) in visited
        ):
            continue

        image[row][col] = replacement
        visited.add((row, col))

        queue.append((row + 1, col))
        queue.append((row - 1, col))
        queue.append((row, col + 1))
        queue.append((row, col - 1))

    return image
