# 🧩 Python Interview Pattern Library

Algorithm templates and patterns. For syntax lookups, see [CHEATSHEET.md](CHEATSHEET.md).

## 📋 Table of Contents

- [Sorting](#sorting)
- [Binary Search](#binary-search)
- [Two Pointers](#two-pointers)
- [Fast & Slow Pointers](#fast--slow-pointers)
- [Sliding Window](#sliding-window)
- [Prefix Sum](#prefix-sum)
- [Monotonic Stack](#monotonic-stack)
- [Intervals](#intervals)
- [Greedy](#greedy)
- [Backtracking](#backtracking)
- [Dynamic Programming](#dynamic-programming)
- [Bit Manipulation](#bit-manipulation)
- [Linked Lists](#linked-lists)
- [Trees](#trees)
- [Binary Search Trees](#binary-search-trees)
- [Matrix](#matrix)
- [Graphs](#graphs)
- [Trie](#trie)
- [Union-Find](#union-find)

---

## Sorting

```python
# Python's built-in sort is Timsort: O(n log n), stable

arr.sort()                      # in-place ascending
sorted(arr)                     # new sorted list

# Custom key
arr.sort(key=len)
arr.sort(key=lambda x: (x[0], -x[1]))   # multi-criteria: x[0] asc, x[1] desc

# Sort dict by values
sorted(d.items(), key=lambda x: x[1])

# Top-k by frequency
from collections import Counter
top_k = [item for item, _ in Counter(arr).most_common(k)]
```

**When to reach for it:** any problem that says "sorted" or where order unlocks a two-pointer / greedy / binary-search approach.

---

## Binary Search

### The 2 Essential Patterns

**Pattern 1: `while left <= right`** — exact target or best candidate
**Pattern 2: `while left < right`** — boundary where condition flips

### Pattern 1 — Classic Binary Search

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**Characteristics:**
- Both pointers move past mid: `left = mid + 1`, `right = mid - 1`
- Loop continues while valid range exists; ends with `left > right`
- Track `result` separately if you want the best candidate (not exact match)

**Common problems:** standard search, rotated sorted array, find largest X where condition holds, square root, TimeMap.

### Pattern 2 — Find Boundary

```python
def find_boundary(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if condition(nums[mid]):
            right = mid          # mid might be the answer — keep it
        else:
            left = mid + 1       # mid is not the answer — skip it
    return left                  # left == right is the answer
```

**Characteristics:**
- `right = mid` (never `mid - 1`) and `left = mid + 1` (never `mid` — would infinite-loop)
- Loop ends when `left == right` — no separate result tracking

**Common problems:** find min in rotated sorted array, first/last occurrence, insertion position, Koko eating bananas, capacity to ship packages.

### Find First / Last Occurrence (Pattern 1 with tracking)

```python
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1            # keep searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

def find_last(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1             # keep searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result
```

Or use `bisect.bisect_left` / `bisect.bisect_right` if you don't need to write it yourself.

### Binary Search on Answer

When the question is "find the minimum/maximum X such that condition(X) holds", binary-search over the answer space:

```python
def min_eating_speed(piles, h):
    def can_finish(k):
        return sum((p + k - 1) // k for p in piles) <= h

    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Two Pointers

```python
# Two Sum in sorted array
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target: return [left, right]
        elif s < target: left += 1
        else: right -= 1
    return []

# Remove duplicates from sorted array (in-place)
def remove_duplicates(nums):
    if not nums: return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1

# Palindrome check
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]: return False
        left += 1
        right -= 1
    return True

# Three Sum — sort, then for each pivot use two pointers
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]: continue   # skip dup pivot
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]: left += 1
                while left < right and nums[right] == nums[right - 1]: right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result

# Container with most water
def max_area(height):
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        best = max(best, min(height[left], height[right]) * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return best
```

---

## Fast & Slow Pointers

Floyd's tortoise-and-hare. Used whenever you need to detect a cycle, find the middle, or find a duplicate in a "function graph".

```python
# Detect cycle in linked list
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast: return True
    return False

# Find start of cycle
def detect_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast: break
    else:
        return None
    # Reset one pointer to head; both move one step at a time
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow

# Find middle of linked list (returns 2nd middle when even length)
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

# Find duplicate in array [1..n] (values as next-pointers)
def find_duplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast: break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow
```

---

## Sliding Window

```python
# Fixed-size window — max sum of k consecutive
def max_sum_subarray(nums, k):
    window_sum = sum(nums[:k])
    best = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        best = max(best, window_sum)
    return best

# Variable-size window — longest substring without repeats
def longest_unique_substring(s):
    seen = set()
    left = best = 0
    for right, c in enumerate(s):
        while c in seen:
            seen.remove(s[left])
            left += 1
        seen.add(c)
        best = max(best, right - left + 1)
    return best

# Minimum window substring containing all chars of t
def min_window(s, t):
    from collections import Counter
    need = Counter(t)
    missing = len(t)
    left = 0
    best_start, best_len = 0, float('inf')

    for right, c in enumerate(s):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1

        if missing == 0:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if right - left + 1 < best_len:
                best_start, best_len = left, right - left + 1
            need[s[left]] += 1
            missing += 1
            left += 1

    return "" if best_len == float('inf') else s[best_start:best_start + best_len]

# Longest substring with at most k distinct characters
def longest_k_distinct(s, k):
    from collections import defaultdict
    count = defaultdict(int)
    left = best = 0
    for right, c in enumerate(s):
        count[c] += 1
        while len(count) > k:
            count[s[left]] -= 1
            if count[s[left]] == 0:
                del count[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best
```

---

## Prefix Sum

Convert range-sum queries from O(n) to O(1) after an O(n) preprocess.

```python
# 1D prefix sum
def build_prefix(nums):
    prefix = [0] * (len(nums) + 1)
    for i, x in enumerate(nums):
        prefix[i + 1] = prefix[i] + x
    return prefix

# Range sum [l, r] inclusive
range_sum = prefix[r + 1] - prefix[l]

# Subarray sum equals K — count subarrays with sum == k
def subarray_sum(nums, k):
    from collections import defaultdict
    count = 0
    prefix_count = defaultdict(int)
    prefix_count[0] = 1                # empty prefix
    running = 0
    for x in nums:
        running += x
        count += prefix_count[running - k]
        prefix_count[running] += 1
    return count

# 2D prefix sum — sum of any submatrix in O(1)
def build_2d_prefix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    for r in range(rows):
        for c in range(cols):
            prefix[r + 1][c + 1] = (matrix[r][c]
                                    + prefix[r][c + 1]
                                    + prefix[r + 1][c]
                                    - prefix[r][c])
    return prefix

# Sum of submatrix from (r1, c1) to (r2, c2) inclusive
def submatrix_sum(prefix, r1, c1, r2, c2):
    return (prefix[r2 + 1][c2 + 1]
            - prefix[r1][c2 + 1]
            - prefix[r2 + 1][c1]
            + prefix[r1][c1])
```

---

## Monotonic Stack

A stack maintained in monotonic (increasing or decreasing) order. Perfect for "next greater / smaller element" style problems in O(n).

```python
# Next Greater Element — for each index, next index with a larger value
def next_greater(nums):
    result = [-1] * len(nums)
    stack = []                          # stores indices, values decreasing
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            result[stack.pop()] = x
        stack.append(i)
    return result

# Daily Temperatures — wait days until warmer
def daily_temperatures(temps):
    result = [0] * len(temps)
    stack = []
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            result[j] = i - j
        stack.append(i)
    return result

# Largest Rectangle in Histogram
def largest_rectangle(heights):
    stack = []                          # indices, heights increasing
    best = 0
    for i, h in enumerate(heights + [0]):     # sentinel flushes the stack
        while stack and heights[stack[-1]] > h:
            top = stack.pop()
            width = i if not stack else i - stack[-1] - 1
            best = max(best, heights[top] * width)
        stack.append(i)
    return best

# Sliding Window Maximum — monotonic deque
def max_sliding_window(nums, k):
    from collections import deque
    dq = deque()                        # indices, nums[dq] decreasing
    result = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:    # drop out-of-window
            dq.popleft()
        while dq and nums[dq[-1]] < x:  # maintain decreasing
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

---

## Intervals

Almost always: **sort by start**, then sweep.

```python
# Merge overlapping intervals
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

# Insert interval into sorted, non-overlapping list
def insert(intervals, new):
    result = []
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < new[0]:
        result.append(intervals[i]); i += 1
    while i < n and intervals[i][0] <= new[1]:
        new = [min(new[0], intervals[i][0]), max(new[1], intervals[i][1])]
        i += 1
    result.append(new)
    while i < n:
        result.append(intervals[i]); i += 1
    return result

# Meeting Rooms II — minimum rooms needed
def min_meeting_rooms(intervals):
    import heapq
    intervals.sort(key=lambda x: x[0])
    heap = []                                # end times of ongoing meetings
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)              # reuse a room
        heapq.heappush(heap, end)
    return len(heap)

# Non-overlapping intervals — minimum removals (greedy by end time)
def erase_overlap_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count = 0
    end = float('-inf')
    for s, e in intervals:
        if s >= end:
            end = e
        else:
            count += 1
    return count
```

---

## Greedy

Make the locally optimal choice at each step. Only works when the problem has the greedy-choice property — usually you need a sorting key or a clear invariant.

```python
# Jump Game — can you reach the end?
def can_jump(nums):
    farthest = 0
    for i, x in enumerate(nums):
        if i > farthest: return False
        farthest = max(farthest, i + x)
    return True

# Jump Game II — minimum jumps to reach end
def jump(nums):
    jumps = current_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps

# Gas Station — find starting index to complete circuit, or -1
def can_complete_circuit(gas, cost):
    if sum(gas) < sum(cost): return -1
    tank = start = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start

# Activity selection (classic greedy) — see "Non-overlapping intervals" above
```

---

## Backtracking

```python
# Generic template
def backtrack(state):
    if is_goal(state):
        result.append(state[:])     # copy! state is mutated
        return
    for choice in get_choices(state):
        if is_valid(choice, state):
            make_choice(choice, state)
            backtrack(state)
            undo_choice(choice, state)

# Permutations
def permutations(nums):
    result = []
    def bt(current, used):
        if len(current) == len(nums):
            result.append(current[:])
            return
        for i, x in enumerate(nums):
            if used[i]: continue
            used[i] = True
            current.append(x)
            bt(current, used)
            current.pop()
            used[i] = False
    bt([], [False] * len(nums))
    return result

# Subsets (power set)
def subsets(nums):
    result = []
    def bt(start, current):
        result.append(current[:])
        for i in range(start, len(nums)):
            current.append(nums[i])
            bt(i + 1, current)
            current.pop()
    bt(0, [])
    return result

# Combinations — choose k from n
def combine(n, k):
    result = []
    def bt(start, current):
        if len(current) == k:
            result.append(current[:])
            return
        for i in range(start, n + 1):
            current.append(i)
            bt(i + 1, current)
            current.pop()
    bt(1, [])
    return result

# Combination Sum (with repetition allowed)
def combination_sum(candidates, target):
    result = []
    candidates.sort()
    def bt(start, current, remaining):
        if remaining == 0:
            result.append(current[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining: break
            current.append(candidates[i])
            bt(i, current, remaining - candidates[i])    # i, not i+1 — reuse allowed
            current.pop()
    bt(0, [], target)
    return result

# Word Search in grid
def exist(board, word):
    rows, cols = len(board), len(board[0])
    def dfs(r, c, i):
        if i == len(word): return True
        if (r < 0 or r >= rows or c < 0 or c >= cols
                or board[r][c] != word[i]):
            return False
        board[r][c] = '#'                    # mark visited
        found = (dfs(r+1, c, i+1) or dfs(r-1, c, i+1)
                 or dfs(r, c+1, i+1) or dfs(r, c-1, i+1))
        board[r][c] = word[i]                # restore
        return found
    return any(dfs(r, c, 0) for r in range(rows) for c in range(cols))

# N-Queens — use three sets for O(1) attack checks
def solve_n_queens(n):
    result = []
    cols, diag1, diag2 = set(), set(), set()   # diag1: r-c, diag2: r+c
    queens = []

    def bt(row):
        if row == n:
            board = ['.' * c + 'Q' + '.' * (n - c - 1) for c in queens]
            result.append(board)
            return
        for c in range(n):
            if c in cols or (row - c) in diag1 or (row + c) in diag2:
                continue
            cols.add(c); diag1.add(row - c); diag2.add(row + c)
            queens.append(c)
            bt(row + 1)
            queens.pop()
            cols.discard(c); diag1.discard(row - c); diag2.discard(row + c)

    bt(0)
    return result
```

---

## Dynamic Programming

### Two flavors

- **Top-down (memoization):** write the recurrence as a recursive function, cache results.
- **Bottom-up (tabulation):** fill a `dp` table iteratively from base cases.

### Memoization — cleanest with `@lru_cache`

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1: return n
    return fib(n - 1) + fib(n - 2)

# AVOID the mutable-default-argument anti-pattern:
#   def fib(n, memo={}):  ← cache leaks across calls!
```

### Tabulation

```python
def fib_tab(n):
    if n < 2: return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

### Kadane's Algorithm (Maximum Subarray)

```python
def max_subarray(nums):
    best = current = nums[0]
    for x in nums[1:]:
        current = max(x, current + x)
        best = max(best, current)
    return best
```

### House Robber (no two adjacent)

```python
def rob(nums):
    prev2 = prev1 = 0
    for x in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + x)
    return prev1
```

### Coin Change (min coins to make amount)

```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

### Longest Increasing Subsequence — O(n log n)

```python
def length_of_LIS(nums):
    import bisect
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
```

### Longest Common Subsequence

```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

### Edit Distance (Levenshtein)

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],         # delete
                                   dp[i][j - 1],         # insert
                                   dp[i - 1][j - 1])     # replace
    return dp[m][n]
```

### 0/1 Knapsack

```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][capacity]
```

### Unbounded Knapsack (Coin Change II — number of ways)

```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:                    # outer loop on coins → counts combinations, not permutations
        for i in range(c, amount + 1):
            dp[i] += dp[i - c]
    return dp[amount]
```

---

## Bit Manipulation

```python
# Basic operations
n & 1                    # check if odd
n | (1 << i)            # set ith bit
n & ~(1 << i)           # clear ith bit
n ^ (1 << i)            # flip ith bit
(n >> i) & 1            # get ith bit

# Tricks
n & (n - 1)             # remove rightmost set bit
n & -n                  # isolate rightmost set bit
n != 0 and n & (n - 1) == 0   # check if power of 2

# Count set bits
def count_bits(n):
    count = 0
    while n:
        n &= n - 1      # remove rightmost set bit
        count += 1
    return count

bin(n).count('1')       # built-in alternative

# XOR properties
# a ^ a == 0, a ^ 0 == a, commutative, associative

# Find single number (all others appear twice)
def single_number(nums):
    result = 0
    for x in nums:
        result ^= x
    return result

# Enumerate all subsets via bitmask
def all_subsets(nums):
    n = len(nums)
    return [[nums[i] for i in range(n) if mask & (1 << i)]
            for mask in range(1 << n)]
```

---

## Linked Lists

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Dummy node pattern — use when the head itself might change
def example(head):
    dummy = ListNode(0, head)
    # ... manipulate dummy.next ...
    return dummy.next

# Reverse a linked list (iterative)
def reverse_list(head):
    prev, curr = None, head
    while curr:
        curr.next, prev, curr = prev, curr, curr.next
    return prev

# Reverse (recursive)
def reverse_list_rec(head):
    if not head or not head.next: return head
    new_head = reverse_list_rec(head.next)
    head.next.next = head
    head.next = None
    return new_head

# Merge two sorted lists
def merge_two_lists(l1, l2):
    dummy = tail = ListNode()
    while l1 and l2:
        if l1.val <= l2.val:
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next
    tail.next = l1 or l2
    return dummy.next

# Remove Nth node from end (one-pass via two pointers)
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1): fast = fast.next
    while fast:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next

# Reorder list: L0 → Ln → L1 → Ln-1 → ...
def reorder_list(head):
    # 1. find middle
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    # 2. reverse second half
    prev, curr = None, slow.next
    slow.next = None
    while curr:
        curr.next, prev, curr = prev, curr, curr.next
    # 3. merge
    first, second = head, prev
    while second:
        first.next, second.next, first, second = second, first.next, first.next, second.next
```

Cycle detection lives in [Fast & Slow Pointers](#fast--slow-pointers).

---

## Trees

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Recursive traversals
def inorder(root):
    result = []
    def go(node):
        if not node: return
        go(node.left); result.append(node.val); go(node.right)
    go(root)
    return result

# Preorder / postorder — swap the order of the three lines above.

# Iterative inorder
def inorder_iter(root):
    result, stack, curr = [], [], root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result

# Level order (BFS)
def level_order(root):
    if not root: return []
    from collections import deque
    queue = deque([root])
    result = []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result

# Max depth
def max_depth(root):
    if not root: return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Is balanced — return (height, balanced) so it's one pass
def is_balanced(root):
    def check(node):
        if not node: return 0, True
        lh, lb = check(node.left)
        rh, rb = check(node.right)
        balanced = lb and rb and abs(lh - rh) <= 1
        return 1 + max(lh, rh), balanced
    return check(root)[1]

# Diameter — uses nonlocal
def diameter_of_binary_tree(root):
    diameter = 0
    def depth(node):
        nonlocal diameter
        if not node: return 0
        l, r = depth(node.left), depth(node.right)
        diameter = max(diameter, l + r)
        return 1 + max(l, r)
    depth(root)
    return diameter

# Lowest Common Ancestor (binary tree)
def lca(root, p, q):
    if not root or root == p or root == q: return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right: return root
    return left or right

# Serialize / deserialize (preorder with None markers)
def serialize(root):
    vals = []
    def go(node):
        if not node:
            vals.append('#'); return
        vals.append(str(node.val))
        go(node.left); go(node.right)
    go(root)
    return ' '.join(vals)

def deserialize(data):
    vals = iter(data.split())
    def go():
        v = next(vals)
        if v == '#': return None
        node = TreeNode(int(v))
        node.left = go()
        node.right = go()
        return node
    return go()
```

---

## Binary Search Trees

In a BST, in-order traversal yields a sorted sequence — that's the key invariant.

```python
# Validate BST
def is_valid_bst(root):
    def check(node, lo, hi):
        if not node: return True
        if not (lo < node.val < hi): return False
        return check(node.left, lo, node.val) and check(node.right, node.val, hi)
    return check(root, float('-inf'), float('inf'))

# Insert
def insert(root, val):
    if not root: return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    return root

# Delete (handles 0/1/2 children)
def delete(root, key):
    if not root: return None
    if key < root.val:
        root.left = delete(root.left, key)
    elif key > root.val:
        root.right = delete(root.right, key)
    else:
        if not root.left: return root.right
        if not root.right: return root.left
        # find in-order successor (smallest in right subtree)
        succ = root.right
        while succ.left: succ = succ.left
        root.val = succ.val
        root.right = delete(root.right, succ.val)
    return root

# Kth smallest — iterative in-order, stop at k
def kth_smallest(root, k):
    stack, curr = [], root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        k -= 1
        if k == 0: return curr.val
        curr = curr.right

# LCA in BST — O(h) using ordering
def lca_bst(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
```

---

## Matrix

```python
# Standard 4-directional neighbors
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Rotate 90° clockwise (in-place)
def rotate(matrix):
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse each row
    for row in matrix:
        row.reverse()

# One-liner (new matrix)
rotated = [list(row) for row in zip(*matrix[::-1])]

# Spiral traversal
def spiral_order(matrix):
    result = []
    while matrix:
        result += matrix.pop(0)                       # top row
        matrix = [list(row) for row in zip(*matrix)][::-1]  # rotate ccw
    return result

# Flood fill / number of islands (DFS)
def num_islands(grid):
    if not grid: return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols
                or grid[r][c] != '1'):
            return
        grid[r][c] = '0'                              # mark visited
        for dr, dc in DIRS:
            dfs(r + dr, c + dc)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count

# Set matrix zeroes in place (use first row/col as marker)
def set_zeroes(matrix):
    rows, cols = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][c] == 0 for c in range(cols))
    first_col_zero = any(matrix[r][0] == 0 for r in range(rows))

    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][c] == 0:
                matrix[r][0] = matrix[0][c] = 0

    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0

    if first_row_zero:
        for c in range(cols): matrix[0][c] = 0
    if first_col_zero:
        for r in range(rows): matrix[r][0] = 0
```

---

## Graphs

```python
# Adjacency list — usually a dict
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    ...
}

# DFS recursive
def dfs(graph, start, visited=None):
    if visited is None: visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# DFS iterative
def dfs_iter(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited: continue
        visited.add(node)
        stack.extend(n for n in graph[node] if n not in visited)
    return visited

# BFS — shortest path in unweighted graph
def bfs(graph, start):
    from collections import deque
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for n in graph[node]:
            if n not in visited:
                visited.add(n)
                queue.append(n)
    return visited

# Topological Sort (Kahn's algorithm)
def topological_sort(graph):
    from collections import defaultdict, deque
    # Initialize in-degree for every node — including those only seen as values
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree.setdefault(neighbor, 0)
            in_degree[neighbor] += 1

    queue = deque(n for n, d in in_degree.items() if d == 0)
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for n in graph.get(node, []):
            in_degree[n] -= 1
            if in_degree[n] == 0:
                queue.append(n)
    return result if len(result) == len(in_degree) else []   # [] = cycle

# Dijkstra's (non-negative weights)
def dijkstra(graph, start):
    import heapq
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > distances[node]: continue              # stale entry
        for neighbor, weight in graph[node].items():
            nd = d + weight
            if nd < distances[neighbor]:
                distances[neighbor] = nd
                heapq.heappush(pq, (nd, neighbor))
    return distances

# Cycle in directed graph — DFS with 3 colors
def has_cycle_directed(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        color[node] = GRAY
        for n in graph[node]:
            if color.get(n, WHITE) == GRAY: return True
            if color.get(n, WHITE) == WHITE and dfs(n): return True
        color[node] = BLACK
        return False

    return any(color[n] == WHITE and dfs(n) for n in graph)

# Bellman-Ford (handles negative weights, detects negative cycles)
def bellman_ford(edges, n, start):
    dist = [float('inf')] * n
    dist[start] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    # One more pass — if anything still relaxes, there's a negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None
    return dist
```

---

## Trie

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            node = node.children.setdefault(c, TrieNode())
        node.is_end = True

    def search(self, word):
        node = self._traverse(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._traverse(prefix) is not None

    def _traverse(self, s):
        node = self.root
        for c in s:
            if c not in node.children: return None
            node = node.children[c]
        return node
```

Common use cases: prefix queries, autocomplete, word search in grid (combine with DFS), longest common prefix.

---

## Union-Find

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return False                       # already connected
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

# Cycle in undirected graph
def has_cycle_undirected(edges, n):
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True
    return False

# Number of connected components
def count_components(n, edges):
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.components
```

**When to reach for Union-Find:** connectivity queries, dynamic component counting, cycle detection in undirected graphs, Kruskal's MST.
