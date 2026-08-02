from collections import deque


def is_valid_course_schedule(n: int, prerequisites: list[list[int]]) -> bool:
    graph = {i: [] for i in range(n)}
    in_degree = {i: 0 for i in range(n)}

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque()

    for key, value in in_degree.items():
        if value == 0:
            queue.append(key)

    courses_taken = 0

    while queue:
        current_course = queue.popleft()
        courses_taken += 1

        for dependent_course in graph[current_course]:
            in_degree[dependent_course] -= 1

            if in_degree[dependent_course] == 0:
                queue.append(dependent_course)

    return courses_taken == n
