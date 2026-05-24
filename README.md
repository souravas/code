# 🐍 Python Coding Interview Reference

Two companion documents for coding interview prep:

| File | Purpose | When to read |
| --- | --- | --- |
| **[CHEATSHEET.md](CHEATSHEET.md)** | Pure Python syntax and idioms | The 30 seconds before a round — *"what does `bisect_left` return again?"* |
| **[PATTERNS.md](PATTERNS.md)** | Algorithm templates and techniques | The night before — *"how does the monotonic-stack template go?"* |

---

## What's in CHEATSHEET.md

Syntax-only, one-liners, no algorithms.

- Basic types: integers, strings, booleans
- Built-in data structures: lists, sets, dicts, tuples, stacks, deques
- Node skeletons for linked lists and trees
- `collections`: Counter, defaultdict, deque
- `heapq`, `math`, `itertools`, `bisect`, `functools`
- Comprehensions, generators, lambdas
- Scope (`global` / `nonlocal`)
- Classes (essentials only — `__init__`, `__str__`, `__eq__`, `__lt__`, `@property`, inheritance)
- Common idioms (`enumerate`, `zip`, unpacking, transpose, etc.)
- Performance tips and Big-O reference

## What's in PATTERNS.md

Algorithm templates — full implementations you can adapt.

- **Searching:** Binary Search (both patterns), Binary Search on Answer
- **Two-pointer family:** Two Pointers, Fast & Slow Pointers, Sliding Window
- **Array tricks:** Prefix Sum (1D & 2D), Monotonic Stack, Intervals
- **Strategies:** Greedy, Backtracking, Dynamic Programming
- **Bits:** Bit Manipulation
- **Data structures:** Linked Lists, Trees, BSTs, Matrix, Graphs, Trie, Union-Find

---

## Interview-Day Tips

1. **Clarify the problem.** Input/output format, edge cases, assumptions.
2. **Brute force first, then optimize.** A correct slow solution beats a broken fast one.
3. **Think out loud.** Walk through your approach and examples before coding.
4. **Test your code.** Walk through with examples; check off-by-one errors and empty inputs.
5. **State complexity.** Always analyze and discuss time/space trade-offs.

### Patterns to drill

Two pointers · Sliding window · Binary search · DFS/BFS · Dynamic programming · Backtracking · Monotonic stack · Prefix sum · Intervals · Union-Find

---

_Happy coding! 🚀_
