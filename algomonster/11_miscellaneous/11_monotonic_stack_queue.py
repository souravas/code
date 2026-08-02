def mono_stack(insert_entries):
    # This implementation maintains a decreasing monotonic stack.

    stack = []
    for entry in insert_entries:
        # Pop smaller (or equal) values so stack stays decreasing.
        while stack and stack[-1] <= entry:
            stack.pop()
            # Do something with the popped item here
        stack.append(entry)
