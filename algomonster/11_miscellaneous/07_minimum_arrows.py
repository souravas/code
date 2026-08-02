def find_min_arrow_shots(points: list[list[int]]) -> int:
    if not points:
        return 0

    points.sort(key=lambda x: x[1])

    arrows = 1
    last_shot = points[0][1]

    for start, end in points[1:]:
        if start > last_shot:
            arrows += 1
            last_shot = end

    return arrows
