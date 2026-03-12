class Edge:
    def __init__(self, weight, a, b):
        self.weight = weight
        self.a = a
        self.b = b


def minimum_spanning_tree(n: int, edges) -> int:
    edges.sort(key=lambda edge: edge.weight)

    dsu = UnionFind()  # type: ignore
    ret, cnt = 0, 0
    for edge in edges:
        # Check if edges belong to the same set before merging
        if dsu.find(edge.a) != dsu.find(edge.b):
            dsu.union(edge.a, edge.b)
            ret += edge.weight
            cnt += 1
            if cnt == n - 1:
                break
    return ret
