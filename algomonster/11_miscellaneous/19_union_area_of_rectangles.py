def rectangle_area_ii(rectangles: list[list[int]]) -> int | float:
    xs = sorted({x for r in rectangles for x in (r[0], r[2])})
    total = 0

    for xl, xr in zip(xs, xs[1:]):
        width = xr - xl
        if width == 0:
            continue
        # y-intervals of rectangles spanning this vertical strip
        spans = sorted(
            (y1, y2)
            for x1, y1, x2, y2 in rectangles
            if x1 <= xl and x2 >= xr and y1 < y2
        )
        covered = 0
        cur_end = float("-inf")
        for y1, y2 in spans:
            y1 = max(y1, cur_end)
            if y2 > y1:
                covered += y2 - y1
                cur_end = y2
        total += width * covered

    return total
