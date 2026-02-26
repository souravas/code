def dfs(start_index, additional_states):
    if is_leaf(start_index):  # type: ignore # noqa: F821
        return 1
    ans = initial_value  # type: ignore # noqa: F821

    for edge in get_edges(start_index, additional_states):  # type: ignore # noqa: F821
        if additional_states:
            update(additional_states)  # type: ignore # noqa: F821
        ans = aggregate(ans, dfs(start_index + len(edge), additional_states))  # type: ignore # noqa: F821
        if additional_states:
            revert(additional_states)  # type: ignore # noqa: F821

    return ans
