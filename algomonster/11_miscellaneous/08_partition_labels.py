def partition_labels(s: str) -> list[int]:
    last = {ch: i for i, ch in enumerate(s)}

    parts = []
    start = end = 0

    for i, ch in enumerate(s):
        end = max(end, last[ch])
        if i == end:
            parts.append(end - start + 1)
            start = i + 1

    return parts
