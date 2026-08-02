def starting_station(gas: list[int], dist: list[int]) -> int:
    if sum(gas) < sum(dist):
        return -1

    start = 0
    tank = 0
    for i in range(len(gas)):
        tank += gas[i] - dist[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start
