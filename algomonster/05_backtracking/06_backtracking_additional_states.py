result = []


def dfs(start_index, path, additional_states):
    if is_leaf(start_index):
        result.append(path[:])
        return

    for edge in get_edges(start_index, additional_states):
        # prune if needed
        if not is_valid(edge):
            continue

        path.add(edge)
        if additional_states:
            update(additional_states)
        dfs(start_index + len(edge), path, additional_states)
        path.pop()


def is_leaf(index): ...


def get_edges(index, states):
    return ""


def is_valid(edge): ...


def update(states): ...
