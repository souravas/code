def min_cost_to_visit_every_node(graph: list[list[int]]) -> int:
    # set dp array size equal to 2^(number of nodes)
    dp = [[0] * len(graph) for _ in range(1 << len(graph))]

    def f(bitmask, cur):
        # check if we have visited every node
        if bitmask == (1 << len(graph)) - 1:
            return 0
        if dp[bitmask][cur] != 0:
            return dp[bitmask][cur]
        # set to arbitrary large value, edges are only 1000 and 15 nodes so total can never reach 0x3F3F3F3F
        ans = 0x3F3F3F3F
        # loop through all the neighbours for this particular node
        for i in range(len(graph[cur])):
            if (bitmask & (1 << i)) == 0 and graph[cur][i] != 0:
                # if we have not visited this node, call the recursive function and see if we get a new minimum
                ans = min(ans, graph[cur][i] + f((bitmask | (1 << i)), i))
        dp[bitmask][cur] = ans
        return ans

    # set node 0 as visited and start at node 0
    ans = f(1, 0)
    return -1 if ans == 0x3F3F3F3F else ans
