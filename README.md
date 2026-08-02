# 🐍 Python Coding Interview Reference

Two companion documents for coding interview prep:

| File | Purpose | When to read |
| --- | --- | --- |
| **[CHEATSHEET.md](CHEATSHEET.md)** | Pure Python syntax and idioms | The 30 seconds before a round — *"what does `bisect_left` return again?"* |
| **[PATTERNS.md](PATTERNS.md)** | Algorithm templates and techniques | The night before — *"how does the monotonic-stack template go?"* |

---

## What's in CHEATSHEET.md

Syntax-only, one-liners, no algorithms.

- Basic types: integers (base conversion, bit introspection, sentinels), strings, booleans
- Built-in data structures: lists, sets, dicts, tuples, stacks, deques
- Node skeletons for linked lists and trees
- `collections`: Counter, defaultdict, OrderedDict, deque
- `heapq`, `math`, `string`, `random`, `itertools`, `bisect`, `functools`
- `sortedcontainers` (SortedList / SortedDict / SortedSet)
- Comprehensions, generators, lambdas
- Scope (`global` / `nonlocal`)
- Classes (essentials only — `__init__`, `__str__`, `__eq__`, `__lt__`, `@property`, inheritance)
- Common idioms (`enumerate`, `zip`, unpacking, transpose, argmax, coordinate compression)
- Performance tips and Big-O reference

## What's in PATTERNS.md

Algorithm templates — full implementations you can adapt. Between them, the sections cover every
technique used anywhere in `algomonster/` and `neetcode/`.

- **Sorting:** built-in sort and `cmp_to_key`, merge sort (and the reusable merge step), the O(n²) sorts
- **Searching:** Binary Search (both patterns), Binary Search on Answer
- **Two-pointer family:** Two Pointers, Fast & Slow Pointers, Sliding Window (variable, fixed-window match counter, shrink-on-duplicate)
- **Array tricks:** Prefix Sum (1D & 2D), Hashing, Monotonic Stack, Intervals, Line Sweep
- **Stacks:** monotonic, parsing (RPN, calculator), design (Min Stack), Car Fleet
- **Top-K:** Heap (top-K, k-closest, merge-K, median of stream, streaming kth-largest), Quickselect
- **Strategies:** Divide & Conquer (count-of-smaller, skyline), Greedy, Backtracking, Dynamic Programming
- **DP families:** linear/stairs, grid (incl. solving backwards), dual-sequence, knapsack (0/1, unbounded, bounded), interval, game theory, DAG, tree, bitmask
- **Bits & math:** Bit Manipulation, Math / Number Theory (sieve, nth prime, modpow)
- **Data structures:** Linked Lists, Trees, BSTs, Matrix, Graphs, Trie, Union-Find, Segment Tree
- **Graphs in depth:** DFS/BFS, multi-source and 0-1 BFS, implicit state-space BFS (word ladder, sliding puzzle), topological sort, Dijkstra, Bellman-Ford, MST (Kruskal & Prim)
- **Design:** LRU Cache (OrderedDict + from-scratch DLL versions)

---

## Interview-Day Tips

1. **Clarify the problem.** Input/output format, edge cases, assumptions.
2. **Brute force first, then optimize.** A correct slow solution beats a broken fast one.
3. **Think out loud.** Walk through your approach and examples before coding.
4. **Test your code.** Walk through with examples; check off-by-one errors and empty inputs.
5. **State complexity.** Always analyze and discuss time/space trade-offs.

### Patterns to drill

Two pointers · Sliding window · Binary search · DFS/BFS · Dynamic programming · Backtracking · Monotonic stack · Heap / top-K · Prefix sum · Intervals · Union-Find

---

_Happy coding! 🚀_
