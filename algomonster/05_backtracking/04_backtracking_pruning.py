def dfs(start_index, path):
    if is_leaf(start_index):
        report(path)
        return

    for edge in get_edges(start_index):
        # prune if needed
        if not is_valid(edge):
            continue
        path.add(edge)
        # increase index by variable size, instead of 1
        dfs(start_index + len(edge), path)
        path.pop()


def is_leaf(index): ...


def report(path): ...


def get_edges(index):
    return []


def is_valid(edge): ...
