#   🐍 Python Coding Interview Cheat Sheet

A comprehensive Python reference for coding interviews, optimized for quick lookup and practical use.

## 📋 Table of Contents

- [General Tips](#general-tips)
- [Basic Data Types](#basic-data-types)
  - [Integers](#integers)
  - [Strings](#strings)
  - [Booleans & None](#booleans--none)
- [Data Structures](#data-structures)
  - [Lists](#lists)
  - [Sets](#sets)
  - [Dictionaries](#dictionaries)
  - [Tuples](#tuples)
  - [Stacks](#stacks)
  - [Deques](#deques)
- [Advanced Collections](#advanced-collections)
  - [Counter](#counter)
  - [DefaultDict](#defaultdict)
  - [Heapq (Priority Queue)](#heapq-priority-queue)
- [Algorithms & Techniques](#algorithms--techniques)
  - [Sorting](#sorting)
  - [Binary Search](#binary-search)
  - [Two Pointers](#two-pointers)
  - [Sliding Window](#sliding-window)
  - [Backtracking](#backtracking)
  - [Dynamic Programming](#dynamic-programming)
  - [Bit Manipulation](#bit-manipulation)
- [Advanced Data Structures](#advanced-data-structures)
  - [Trees](#trees)
  - [Graphs](#graphs)
  - [Trie](#trie)
  - [Union-Find](#union-find)
- [Useful Modules](#useful-modules)
  - [Math](#math)
  - [Itertools](#itertools)
  - [Bisect](#bisect)
  - [Functools](#functools)
- [Python Features](#python-features)
  - [Comprehensions](#comprehensions)
  - [Generators](#generators)
  - [Lambda Functions](#lambda-functions)
  - [Classes](#classes)
- [Best Practices](#best-practices)

---

## General Tips

```python
# Preferred comparisons
is None          # preferred
== None          # depends on eq method of object, not preferred

# Truthiness checks
while head:      # while head is not None
if not head:     # if head is None

# Swapping
left, right = right, left

# Check if empty
not stack        # returns True if datastructure is empty
len(stack) == 0  # explicit length check
```

---

## Basic Data Types

### Integers

```python
# Conversion and basic operations
int("123")       # string to int
abs(-5)          # absolute value: 5
2**3             # power: 8 (preferred over math.pow)
divmod(10, 3)    # (3, 1) - quotient and remainder

# Division types
5 / 2            # 2.5 (float division)
5 // 2           # 2 (floor division)
5 % 2            # 1 (modulo)

# Handle negative numbers carefully
-3 // 2          # -2 (rounds towards negative infinity)
int(-3 / 2)      # -1 (rounds towards zero)
import math
math.fmod(-10, 3) # -1.0 (correct modulo towards zero)
```

### Strings

```python
# Strings are immutable - operations return new strings
s = "python is good"
s.split()                    # ['python', 'is', 'good'] (splits by whitespace)
s.split('#')                 # splits by delimiter
''.join(my_list)            # join list elements into string

# Search and replace
s.find('is')                # returns index or -1 if not found
s.find('is', start, end)    # search in substring
s.replace('good', 'great')  # replace all occurrences
s.strip()                   # remove whitespace from both ends
s.strip(' #')               # remove specific characters

# Case operations
s.upper()                   # convert to uppercase
s.lower()                   # convert to lowercase
s.title()                   # Title Case
s.capitalize()              # Capitalize first letter

# Checks
s.isalpha()                 # only letters (a-z, A-Z)
s.isdigit()                 # only digits (0-9)
s.isalnum()                 # letters and digits
s.isspace()                 # only whitespace
s.islower()                 # all lowercase
s.isupper()                 # all uppercase
s.startswith('py')          # starts with substring
s.endswith('od')            # ends with substring

# String manipulation
"abc" * 3                   # 'abcabcabc'
len(s)                      # string length
s[0:3]                      # substring (slicing)
sorted(s)                   # returns sorted list of characters
''.join(sorted(s))          # sort characters in string

# ASCII operations
ord('a')                    # 97 (ASCII value)
chr(97)                     # 'a' (character from ASCII)

# Character frequency array (for lowercase letters)
arr = [0] * 26
for char in s.lower():
    if char.isalpha():
        arr[ord(char) - ord('a')] += 1

# Format strings
name, age = "Alice", 25
f"Hello {name}, you are {age}"          # f-strings (preferred)
"Hello {}, you are {}".format(name, age) # .format method
"Hello %s, you are %d" % (name, age)    # % formatting
```

### Booleans & None

```python
# Boolean operations
True and False           # False
True or False           # True
not True                # False

# Truthiness in Python
bool([])                # False (empty list)
bool([1])               # True (non-empty list)
bool("")                # False (empty string)
bool("a")               # True (non-empty string)
bool(0)                 # False
bool(42)                # True

# None checks
x is None               # preferred
x == None               # not preferred
x is not None           # preferred
x != None               # not preferred
```

---

## Data Structures

### Lists

```python
# Creation
[1] * n                     # [1, 1, 1, 1, 1]
[1, 2, 3] * 2              # [1, 2, 3, 1, 2, 3]
list("sourav")             # ['s', 'o', 'u', 'r', 'a', 'v']
list(range(5))             # [0, 1, 2, 3, 4]

# 2D arrays
matrix = [[0] * cols for _ in range(rows)]  # correct way
matrix = [[0] * cols] * rows                # wrong! creates shallow copies

# Adding elements
arr.append(x)              # add to end
arr.insert(i, x)           # insert at index i
arr.extend([1, 2, 3])      # add multiple elements
arr1 + arr2                # concatenate lists
[*arr1, *arr2]             # unpack and concatenate

# Removing elements
arr.remove(x)              # remove first occurrence of x (raises ValueError if not found)
arr.pop()                  # remove and return last element
arr.pop(i)                 # remove and return element at index i
del arr[i]                 # delete element at index i
arr.clear()                # remove all elements

# Copying
arr.copy()                 # shallow copy
arr[:]                     # shallow copy
import copy
copy.deepcopy(arr)         # deep copy

# Sorting and reversing
arr.sort()                 # in-place sort (ascending)
arr.sort(reverse=True)     # in-place sort (descending)
sorted(arr)                # returns new sorted list
arr.reverse()              # in-place reverse
list(reversed(arr))        # returns new reversed list

# Custom sorting
arr.sort(key=len)          # sort by length
arr.sort(key=lambda x: (x[0], -x[1]))  # multiple criteria

# Searching
arr.index(x)               # index of first occurrence (raises ValueError if not found)
arr.count(x)               # count occurrences
x in arr                   # check if element exists

# Useful operations
len(arr)                   # length
min(arr)                   # minimum element
max(arr)                   # maximum element
sum(arr)                   # sum of elements
all(arr)                   # True if all elements are truthy
any(arr)                   # True if any element is truthy

# Slicing (works on strings too)
arr[start:end]             # elements from start to end-1
arr[start:]                # elements from start to end
arr[:end]                  # elements from beginning to end-1
arr[start:end:step]        # elements from start to end-1 with step
arr[::-1]                  # reverse the list
```

### Sets

```python
# Creation
s = set()                  # empty set
s = {1, 2, 3}             # set with elements
s = set([1, 2, 2, 3])     # {1, 2, 3} from list

# Adding elements
s.add(x)                   # add single element
s.update([1, 2, 3])        # add multiple elements

# Removing elements
s.remove(x)                # remove x (raises KeyError if not found)
s.discard(x)               # remove x (no error if not found)
s.pop()                    # remove and return arbitrary element
s.clear()                  # remove all elements

# Set operations
s1 | s2                    # union
s1 & s2                    # intersection
s1 - s2                    # difference
s1 ^ s2                    # symmetric difference
s1.issubset(s2)           # check if s1 is subset of s2
s1.issuperset(s2)         # check if s1 is superset of s2

# Useful operations
len(s)                     # number of elements
x in s                     # membership test (O(1))
list(s)                    # convert to list
```

### Dictionaries

```python
# Creation
d = {}                     # empty dict
d = {'a': 1, 'b': 2}      # dict with elements
d = dict(a=1, b=2)        # using dict constructor
d = dict([('a', 1), ('b', 2)])  # from list of tuples

# Accessing elements
d['key']                   # get value (raises KeyError if not found)
d.get('key')              # get value (returns None if not found)
d.get('key', default)     # get value with default
d.setdefault('key', default)  # get value, set default if key doesn't exist

# Adding/updating elements
d['key'] = value          # set value
d.update({'c': 3, 'd': 4})  # update with another dict

# Removing elements
del d['key']              # delete key (raises KeyError if not found)
d.pop('key')              # remove and return value
d.pop('key', default)     # remove and return value with default
d.popitem()               # remove and return arbitrary (key, value) pair
d.clear()                 # remove all elements

# Dictionary views
d.keys()                  # dict_keys object
d.values()                # dict_values object
d.items()                 # dict_items object (key-value pairs)

# Iteration
for key in d:             # iterate over keys
for value in d.values():  # iterate over values
for key, value in d.items():  # iterate over key-value pairs

# Dictionary operations
len(d)                    # number of key-value pairs
'key' in d               # check if key exists
'key' not in d           # check if key doesn't exist

# Merging dictionaries
{**d1, **d2}             # merge two dicts (Python 3.5+)
d1 | d2                  # merge two dicts (Python 3.9+)

# Dictionary comprehension
{k: v for k, v in pairs}
{k: v**2 for k, v in d.items() if v > 0}
```

### Tuples

```python
# Creation
t = ()                     # empty tuple
t = (1,)                   # single element tuple (comma required)
t = (1, 2, 3)             # multiple elements
t = 1, 2, 3               # tuple packing

# Accessing elements
t[0]                      # first element
t[-1]                     # last element

# Tuple operations
len(t)                    # length
t.count(x)                # count occurrences
t.index(x)                # index of first occurrence
x in t                    # membership test

# Tuple unpacking
a, b, c = (1, 2, 3)       # unpack all elements
a, *b, c = (1, 2, 3, 4)   # a=1, b=[2,3], c=4

# Tuples as dictionary keys (since they're immutable)
d = {(0, 1): 'value'}
```

### Stacks

```python
# Using list as stack (LIFO - Last In, First Out)
stack = []

# Stack operations
stack.append(x)           # push
x = stack.pop()           # pop (raises IndexError if empty)
x = stack[-1]             # peek (top element)

# Check if empty
if not stack:             # stack is empty
if stack:                 # stack is not empty

# Stack with error handling
try:
    x = stack.pop()
except IndexError:
    print("Stack is empty")
```

### Deques

```python
from collections import deque

# Creation
dq = deque()              # empty deque
dq = deque([1, 2, 3])     # deque from list
dq = deque(maxlen=5)      # bounded deque

# Adding elements
dq.append(x)              # add to right
dq.appendleft(x)          # add to left
dq.extend([1, 2, 3])      # extend right
dq.extendleft([1, 2, 3])  # extend left (order reversed)

# Removing elements
x = dq.pop()              # remove from right
x = dq.popleft()          # remove from left

# Other operations
dq.rotate(n)              # rotate n steps to right (negative for left)
dq.reverse()              # reverse in-place
dq.clear()                # remove all elements

# Useful for BFS and sliding window problems
```

---

## Advanced Collections

### Counter

```python
from collections import Counter

# Creation
c = Counter()                    # empty counter
c = Counter([1, 2, 2, 3, 3, 3]) # Counter({3: 3, 2: 2, 1: 1})
c = Counter("hello")             # Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})

# Most common elements
c.most_common()              # all elements, most common first
c.most_common(2)             # top 2 most common
[k for k, v in c.most_common(2)]  # just the keys

# Counter operations
c1 + c2                      # add counts
c1 - c2                      # subtract counts
c1 & c2                      # intersection (minimum counts)
c1 | c2                      # union (maximum counts)

# Other methods
c.elements()                 # iterator over elements (with repeats)
c.subtract(other)            # subtract counts in-place
c.update(other)              # add counts in-place

# Convert to list/set
list(c.elements())           # list with repeats
list(c.keys())               # unique elements
```

### DefaultDict

```python
from collections import defaultdict

# Creation with different default values
dd = defaultdict(int)        # default value: 0
dd = defaultdict(list)       # default value: []
dd = defaultdict(set)        # default value: set()
dd = defaultdict(str)        # default value: ''
dd = defaultdict(lambda: 'default')  # custom default value

# Usage
dd['new_key']               # automatically creates key with default value
dd.get('key')               # returns None, doesn't create key

# Common pattern for grouping
groups = defaultdict(list)
for item in items:
    groups[get_group(item)].append(item)

# 2D grid with tuples as keys
grid = defaultdict(bool)
grid[(row, col)] = True
```

### Heapq (Priority Queue)

```python
import heapq

# Min heap (default)
heap = []
heapq.heappush(heap, 3)     # push element
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)

min_val = heap[0]           # peek at minimum (don't pop)
min_val = heapq.heappop(heap)  # pop minimum

# Heapify existing list
arr = [3, 1, 4, 1, 5]
heapq.heapify(arr)          # convert to heap in-place

# Push and pop in one operation
heapq.heappushpop(heap, item)  # push item, then pop minimum
heapq.heapreplace(heap, item)  # pop minimum, then push item

# N largest/smallest
heapq.nlargest(k, iterable)
heapq.nsmallest(k, iterable)

# Max heap (use negative values)
max_heap = []
heapq.heappush(max_heap, -x)  # negate when pushing
max_val = -heapq.heappop(max_heap)  # negate when popping

# Priority queue with tuples
pq = []
heapq.heappush(pq, (priority, item))
priority, item = heapq.heappop(pq)

# Custom objects (implement __lt__ or use tuple with priority)
heapq.heappush(pq, (priority, counter, obj))  # counter for tie-breaking
```

---

## Algorithms & Techniques

### Sorting

```python
# Basic sorting
arr.sort()                      # in-place, ascending
arr.sort(reverse=True)          # in-place, descending
sorted(arr)                     # new list, ascending
sorted(arr, reverse=True)       # new list, descending

# Custom sorting
arr.sort(key=lambda x: len(x))  # sort by length
arr.sort(key=lambda x: (x[0], -x[1]))  # multiple criteria

# Sort by multiple keys
students = [('Alice', 85), ('Bob', 90), ('Charlie', 85)]
students.sort(key=lambda x: (-x[1], x[0]))  # by grade desc, then name asc

# Sort dictionary by values
d = {'a': 3, 'b': 1, 'c': 2}
sorted(d.items(), key=lambda x: x[1])  # [('b', 1), ('c', 2), ('a', 3)]

# Get top k elements by frequency
from collections import Counter
counter = Counter(arr)
top_k = [item for item, count in counter.most_common(k)]
# or
top_k = sorted(counter, key=counter.get, reverse=True)[:k]
```

### Binary Search

```python
import bisect

# Built-in binary search
arr = [1, 3, 5, 7, 9]
bisect.bisect_left(arr, 5)    # leftmost position where 5 can be inserted
bisect.bisect_right(arr, 5)   # rightmost position where 5 can be inserted
bisect.insort(arr, 6)         # insert 6 in sorted order

# Manual binary search template
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
    return -1  # not found

# Find first occurrence
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1  # continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

# Find last occurrence
def find_last(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1  # continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

```
#### The 2 Essential Binary Search Patterns
1. You want an exact index or best candidate (≤ or ≥ target)?
    - Use Pattern 1: while left <= right
2. You want a boundary/first position where a condition flips (false → true or vice versa)?
    - Use Pattern 2: while left < right
##### Pattern 1: `while left <= right` (Classic Binary Search)
###### **Use case:** Finding an exact target OR tracking the best candidate seen so far
```python
def binary_search_pattern1(nums, target):
    left = 0
    right = len(nums) - 1
    result = -1  # or "" or None, depending on problem

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid  # Found exact match
        elif nums[mid] < target:
            # Optionally store mid as candidate if needed
            result = mid  # Store best answer so far
            left = mid + 1
        else:
            right = mid - 1

    return result  # Return tracked result or -1 if not found
```
###### **Key characteristics:**
- Both pointers move **past** mid: `left = mid + 1` and `right = mid - 1`
- Loop continues while valid range exists
- Must track result separately if not finding exact match
- After loop ends, pointers have crossed (`left > right`)
###### **Common problems:**
- Standard binary search for a value
- Search in rotated sorted array
- Finding closest value ≤ or ≥ target (TimeMap problem)
- Finding square root, or any "find largest X where condition holds"
##### Pattern 2: `while left < right` (Finding Boundaries)
###### **Use case:** Finding a position/boundary where a condition transitions
```python
def binary_search_pattern2(nums):
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if condition(nums[mid]):  # e.g., nums[mid] < nums[right]
            right = mid      # mid might be the answer, keep it
        else:
            left = mid + 1   # mid is not the answer, skip it

    return left  # left == right, this is your answer
```
###### **Key characteristics:**
- One pointer includes mid: `right = mid` (never `right = mid - 1`)
- Other pointer skips mid: `left = mid + 1` (never `left = mid` to avoid infinite loop)
- Loop stops when `left == right` (converged to one position)
- No need to track result separately, `left` is your answer
###### **Common problems:**
- Find minimum in rotated sorted array
- Find first/last occurrence of target
- Find insertion position (lower_bound/upper_bound)
- Search in 2D matrix
- Koko eating bananas, capacity to ship packages (minimize/maximize problems)

### Two Pointers

```python
# Two Sum in sorted array
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

# Remove duplicates from sorted array
def remove_duplicates(nums):
    if not nums:
        return 0
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
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

# Three Sum
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:  # skip duplicates
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    return result
```

### Sliding Window

```python
# Maximum sum of k consecutive elements
def max_sum_subarray(nums, k):
    if len(nums) < k:
        return 0

    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(len(nums) - k):
        window_sum = window_sum - nums[i] + nums[i + k]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Longest substring without repeating characters
def longest_unique_substring(s):
    char_set = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length

# Minimum window substring
def min_window(s, t):
    from collections import Counter
    need = Counter(t)
    missing = len(t)
    left = start = end = 0

    for right, char in enumerate(s):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1

        if missing == 0:
            while left <= right and need[s[left]] < 0:
                need[s[left]] += 1
                left += 1

            if not end or right - left <= end - start:
                start, end = left, right

            need[s[left]] += 1
            missing += 1
            left += 1

    return s[start:end + 1] if end else ""
```

### Backtracking

```python
# Backtracking template
def backtrack(state):
    if is_goal(state):
        result.append(state[:])  # make a copy
        return

    for choice in get_choices(state):
        if is_valid(choice, state):
            make_choice(choice, state)
            backtrack(state)
            undo_choice(choice, state)

# Generate all permutations
def permutations(nums):
    result = []

    def backtrack(current):
        if len(current) == len(nums):
            result.append(current[:])
            return

        for num in nums:
            if num not in current:
                current.append(num)
                backtrack(current)
                current.pop()

    backtrack([])
    return result

# Generate all subsets
def subsets(nums):
    result = []

    def backtrack(start, current):
        result.append(current[:])

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result

# N-Queens
def solve_n_queens(n):
    result = []
    board = ['.' * n for _ in range(n)]

    def is_safe(row, col):
        for i in range(row):
            if board[i][col] == 'Q':
                return False
            if col - (row - i) >= 0 and board[i][col - (row - i)] == 'Q':
                return False
            if col + (row - i) < n and board[i][col + (row - i)] == 'Q':
                return False
        return True

    def backtrack(row):
        if row == n:
            result.append(board[:])
            return

        for col in range(n):
            if is_safe(row, col):
                board[row] = board[row][:col] + 'Q' + board[row][col+1:]
                backtrack(row + 1)
                board[row] = board[row][:col] + '.' + board[row][col+1:]

    backtrack(0)
    return result
```

### Dynamic Programming

```python
# Fibonacci with memoization
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Fibonacci with tabulation
def fib_tab(n):
    if n <= 2:
        return 1
    dp = [0] * (n + 1)
    dp[1] = dp[2] = 1
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# Coin change
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# Longest Common Subsequence
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

# 0/1 Knapsack
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],  # don't take item
                    dp[i-1][w - weights[i-1]] + values[i-1]  # take item
                )
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][capacity]

# Using functools.lru_cache for memoization
from functools import lru_cache

@lru_cache(maxsize=None)
def dp_function(param1, param2):
    # base case
    if base_condition:
        return base_value

    # recursive case
    return min/max(dp_function(param1', param2') + cost)
```

### Bit Manipulation

```python
# Basic operations
n & 1                    # check if odd (last bit is 1)
n | (1 << i)            # set ith bit
n & ~(1 << i)           # clear ith bit
n ^ (1 << i)            # flip ith bit
(n >> i) & 1            # get ith bit

# Common patterns
n & (n - 1)             # remove rightmost set bit
n & (-n)                # isolate rightmost set bit
n & (n - 1) == 0        # check if power of 2 (and n > 0)

# Count set bits
def count_bits(n):
    count = 0
    while n:
        count += 1
        n &= n - 1      # remove rightmost set bit
    return count

# Or use built-in
bin(n).count('1')       # count set bits

# XOR properties
a ^ a == 0              # anything XOR itself is 0
a ^ 0 == a              # anything XOR 0 is itself
# XOR is commutative and associative

# Find single number (all others appear twice)
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

# Check if two numbers have opposite signs
def opposite_signs(x, y):
    return (x ^ y) < 0

# Swap without temporary variable
a = a ^ b
b = a ^ b
a = a ^ b
```

---

## Advanced Data Structures

### Trees

```python
# Binary Tree Node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Tree traversals
def inorder(root):
    result = []
    def traverse(node):
        if node:
            traverse(node.left)
            result.append(node.val)
            traverse(node.right)
    traverse(root)
    return result

def preorder(root):
    result = []
    def traverse(node):
        if node:
            result.append(node.val)
            traverse(node.left)
            traverse(node.right)
    traverse(root)
    return result

def postorder(root):
    result = []
    def traverse(node):
        if node:
            traverse(node.left)
            traverse(node.right)
            result.append(node.val)
    traverse(root)
    return result

# Level order traversal (BFS)
def level_order(root):
    if not root:
        return []

    from collections import deque
    queue = deque([root])
    result = []

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result

# Tree height/depth
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Check if balanced
def is_balanced(root):
    def check(node):
        if not node:
            return 0, True

        left_height, left_balanced = check(node.left)
        right_height, right_balanced = check(node.right)

        balanced = (left_balanced and right_balanced and
                   abs(left_height - right_height) <= 1)
        height = 1 + max(left_height, right_height)

        return height, balanced

    return check(root)[1]

# Lowest Common Ancestor
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root
    return left or right
```

### Graphs

```python
# Graph representations
# Adjacency List
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Adjacency Matrix
n = 5
adj_matrix = [[0] * n for _ in range(n)]
# adj_matrix[i][j] = 1 if edge exists from i to j

# DFS (Depth-First Search)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start)  # process node

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

    return visited

# DFS iterative
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node)  # process node
            stack.extend(neighbor for neighbor in graph[node]
                        if neighbor not in visited)

    return visited

# BFS (Breadth-First Search)
def bfs(graph, start):
    from collections import deque
    visited = set([start])
    queue = deque([start])

    while queue:
        node = queue.popleft()
        print(node)  # process node

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited

# Topological Sort (Kahn's Algorithm)
def topological_sort(graph):
    from collections import defaultdict, deque

    in_degree = defaultdict(int)
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in graph if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == len(graph) else []

# Dijkstra's Algorithm
def dijkstra(graph, start):
    import heapq
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current_dist > distances[current]:
            continue

        for neighbor, weight in graph[current].items():
            distance = current_dist + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

# Detect cycle in directed graph
def has_cycle_directed(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        if color[node] == GRAY:
            return True
        if color[node] == BLACK:
            return False

        color[node] = GRAY
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True

        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False
```

### Trie

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def delete(self, word):
        def _delete(node, word, index):
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                node.is_end_of_word = False
                return len(node.children) == 0

            char = word[index]
            if char not in node.children:
                return False

            should_delete_child = _delete(node.children[char], word, index + 1)

            if should_delete_child:
                del node.children[char]
                return not node.is_end_of_word and len(node.children) == 0

            return False

        _delete(self.root, word, 0)

# Usage
trie = Trie()
words = ["apple", "app", "application", "apply", "orange"]
for word in words:
    trie.insert(word)

print(trie.search("app"))        # True
print(trie.search("appl"))       # False
print(trie.starts_with("app"))   # True
```

### Union-Find

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def count_components(self):
        return self.components

# Usage for detecting cycles in undirected graph
def has_cycle_undirected(edges, n):
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True
    return False

# Number of islands using Union-Find
def num_islands(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    uf = UnionFind(rows * cols)

    def index(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                        grid[nr][nc] == '1'):
                        uf.union(index(r, c), index(nr, nc))

    return len(set(uf.find(index(r, c))
                  for r in range(rows)
                  for c in range(cols)
                  if grid[r][c] == '1'))
```

---

## Useful Modules

### Math

```python
import math

# Basic functions
math.ceil(3.2)          # 4 (ceiling)
math.floor(3.8)         # 3 (floor)
math.sqrt(16)           # 4.0 (square root)
math.pow(2, 3)          # 8.0 (power)
math.log(8, 2)          # 3.0 (logarithm base 2)
math.log10(100)         # 2.0 (base 10)
math.log(math.e)        # 1.0 (natural log)

# Trigonometry
math.sin(math.pi / 2)   # 1.0
math.cos(0)             # 1.0
math.tan(math.pi / 4)   # 1.0

# Constants
math.pi                 # 3.14159...
math.e                  # 2.71828...
math.inf                # infinity
-math.inf               # negative infinity
math.nan                # not a number

# Useful functions
math.gcd(12, 18)        # 6 (greatest common divisor)
math.factorial(5)       # 120
math.comb(5, 2)         # 10 (combinations)
math.perm(5, 2)         # 20 (permutations)

# Check for special values
math.isnan(x)           # check if NaN
math.isinf(x)           # check if infinite
math.isfinite(x)        # check if finite
```

### Itertools

```python
import itertools

# Infinite iterators
count = itertools.count(5, 2)       # 5, 7, 9, 11, ...
cycle = itertools.cycle([1, 2, 3])  # 1, 2, 3, 1, 2, 3, ...
repeat = itertools.repeat('A', 3)   # 'A', 'A', 'A'

# Combinatorics
list(itertools.permutations([1, 2, 3]))                # all permutations
list(itertools.combinations([1, 2, 3, 4], 2))          # choose 2
list(itertools.combinations_with_replacement([1, 2], 2)) # with repetition
list(itertools.product([1, 2], [3, 4]))                # cartesian product

# Other useful functions
list(itertools.chain([1, 2], [3, 4], [5]))            # flatten
list(itertools.compress('ABCD', [1, 0, 1, 1]))        # filter by mask
list(itertools.dropwhile(lambda x: x < 5, [1,4,6,4,1])) # drop while condition
list(itertools.takewhile(lambda x: x < 5, [1,4,6,4,1])) # take while condition

# Grouping
data = [1, 1, 2, 2, 2, 3, 1, 1]
for key, group in itertools.groupby(data):
    print(key, list(group))
# Output: 1 [1, 1], 2 [2, 2, 2], 3 [3], 1 [1, 1]

# Examples
# Generate all possible passwords of length 3 with digits
passwords = itertools.product('0123456789', repeat=3)
# All permutations of string
perms = itertools.permutations('ABC')
```

### Bisect

```python
import bisect

# Binary search in sorted array
arr = [1, 3, 5, 7, 9, 11]

# Find insertion points
bisect.bisect_left(arr, 5)    # 2 (leftmost position)
bisect.bisect_right(arr, 5)   # 3 (rightmost position)
bisect.bisect(arr, 5)         # same as bisect_right

# Insert in sorted order
bisect.insort_left(arr, 4)    # insert at leftmost position
bisect.insort_right(arr, 4)   # insert at rightmost position
bisect.insort(arr, 4)         # same as insort_right

# Custom key function (Python 3.10+)
import bisect
data = [(1, 'a'), (3, 'c'), (5, 'e'), (7, 'g')]
bisect.bisect_left(data, 4, key=lambda x: x[0])  # search by first element

# Manual implementation for older Python versions
def bisect_left_key(arr, x, key):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if key(arr[mid]) < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

# Use cases
# 1. Maintain sorted list with insertions
sorted_list = []
for item in items:
    bisect.insort(sorted_list, item)

# 2. Find range of equal elements
def find_range(arr, target):
    left = bisect.bisect_left(arr, target)
    right = bisect.bisect_right(arr, target)
    return (left, right) if left < right else (-1, -1)
```

### Functools

```python
import functools

# Memoization with lru_cache
@functools.lru_cache(maxsize=128)
def expensive_function(n):
    # Some expensive computation
    return n * n

# Clear cache
expensive_function.cache_clear()

# Cache info
print(expensive_function.cache_info())

# Reduce function
from functools import reduce
numbers = [1, 2, 3, 4, 5]
sum_all = reduce(lambda x, y: x + y, numbers)      # 15
product = reduce(lambda x, y: x * y, numbers)      # 120

# Partial application
def multiply(x, y):
    return x * y

double = functools.partial(multiply, 2)
print(double(5))  # 10

# Total ordering decorator
@functools.total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def __eq__(self, other):
        return self.grade == other.grade

    def __lt__(self, other):
        return self.grade < other.grade
    # Now you get all comparison operators for free!

# Singledispatch for function overloading
@functools.singledispatch
def process(arg):
    print(f"Processing {arg}")

@process.register
def _(arg: int):
    print(f"Processing integer: {arg}")

@process.register
def _(arg: str):
    print(f"Processing string: {arg}")
```

---

## Python Features

### Comprehensions

```python
# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
nested = [[x * y for y in range(3)] for x in range(3)]

# Dictionary comprehension
word_lengths = {word: len(word) for word in words}
filtered_dict = {k: v for k, v in my_dict.items() if v > 0}

# Set comprehension
unique_squares = {x**2 for x in [-2, -1, 0, 1, 2]}

# Generator expression (memory efficient)
sum_of_squares = sum(x**2 for x in range(1000000))

# Conditional expressions in comprehensions
# Format: expression_if_true if condition else expression_if_false
result = [x if x > 0 else 0 for x in numbers]

# Multiple conditions
filtered = [x for x in range(100)
           if x % 2 == 0
           if x % 3 == 0
           if x > 10]

# Nested loops in comprehensions
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]

# Equivalent to:
flattened = []
for row in matrix:
    for num in row:
        flattened.append(num)
```

### Generators

```python
# Generator function
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Usage
fib = fibonacci()
first_10 = [next(fib) for _ in range(10)]

# Generator with finite sequence
def squares(n):
    for i in range(n):
        yield i ** 2

# Generator expression
squares_gen = (x**2 for x in range(10))

# Benefits: memory efficient, lazy evaluation
# Use when you don't need all values at once

# Send values to generator
def echo():
    while True:
        value = yield
        print(f"Received: {value}")

gen = echo()
next(gen)  # prime the generator
gen.send("Hello")  # send value

# Generator with return value
def generator_with_return():
    yield 1
    yield 2
    return "finished"

try:
    gen = generator_with_return()
    print(next(gen))  # 1
    print(next(gen))  # 2
    print(next(gen))  # raises StopIteration
except StopIteration as e:
    print(e.value)    # "finished"
```

### Lambda Functions

```python
# Basic lambda
add = lambda x, y: x + y
square = lambda x: x**2

# Lambda with conditionals
max_func = lambda x, y: x if x > y else y

# Lambda in higher-order functions
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))
squares = list(map(lambda x: x**2, numbers))

# Sorting with lambda
students = [('Alice', 85), ('Bob', 90), ('Charlie', 78)]
students.sort(key=lambda student: student[1])  # sort by grade

# Multiple arguments
multiply = lambda x, y, z: x * y * z
print(multiply(2, 3, 4))  # 24

# Lambda with default arguments
power = lambda x, n=2: x**n
print(power(5))    # 25
print(power(5, 3)) # 125

# Limitations: only expressions, no statements
# No assignments, no print statements, no return statements
```

### Classes

```python
# Basic class
class Person:
    # Class variable (shared by all instances)
    species = "Homo sapiens"

    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age
        self._protected = "protected"  # convention
        self.__private = "private"     # name mangling

    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"

    def birthday(self):
        self.age += 1

    # String representation
    def __str__(self):
        return f"Person(name='{self.name}', age={self.age})"

    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

    # Comparison operators
    def __eq__(self, other):
        return self.age == other.age

    def __lt__(self, other):
        return self.age < other.age

# Property decorator
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def area(self):
        return 3.14159 * self._radius ** 2

# Class methods and static methods
class MathUtils:
    @staticmethod
    def add(x, y):
        return x + y

    @classmethod
    def create_zero(cls):
        return cls(0, 0)

# Inheritance
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id

    def introduce(self):  # Override parent method
        base_intro = super().introduce()
        return f"{base_intro}. My student ID is {self.student_id}"

# Multiple inheritance
class Mixin1:
    def method1(self):
        return "method1"

class Mixin2:
    def method2(self):
        return "method2"

class Combined(Mixin1, Mixin2):
    pass

# Abstract base class
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

# Data classes (Python 3.7+)
from dataclasses import dataclass, field
from typing import List

@dataclass
class Point:
    x: int
    y: int
    metadata: List[str] = field(default_factory=list)

    def distance_from_origin(self):
        return (self.x**2 + self.y**2)**0.5

# Usage
p = Point(3, 4)
print(p.distance_from_origin())  # 5.0
```

---

## Best Practices

### Time and Space Complexity

```python
# Common complexities to know:
# O(1) - Constant: hash table operations, array access
# O(log n) - Logarithmic: binary search, balanced trees
# O(n) - Linear: single loop, linear search
# O(n log n) - Linearithmic: efficient sorting algorithms
# O(n²) - Quadratic: nested loops, bubble sort
# O(2^n) - Exponential: recursive fibonacci, subset generation
# O(n!) - Factorial: permutation generation

# Space complexity examples:
# O(1) - few variables regardless of input size
# O(n) - space proportional to input size (single array/list)
# O(n²) - 2D matrix of size n×n
```

### Performance Tips

```python
# Use appropriate data structures
# - set/dict for O(1) lookup instead of list O(n)
# - deque for O(1) append/pop from both ends
# - heapq for priority queue operations

# String concatenation
# Bad: O(n²) for multiple concatenations
result = ""
for word in words:
    result += word

# Good: O(n)
result = ''.join(words)

# List operations
# Use list comprehensions when possible (faster than loops)
squares = [x**2 for x in range(10)]  # faster
squares = []
for x in range(10):
    squares.append(x**2)             # slower

# Use built-in functions
max(numbers)        # faster than manual loop
sum(numbers)        # faster than manual loop
any(conditions)     # faster than manual loop
all(conditions)     # faster than manual loop
```

### Error Handling

```python
# Specific exception handling
try:
    result = risky_operation()
except ValueError as e:
    print(f"Value error: {e}")
except KeyError as e:
    print(f"Key error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
else:
    print("No exception occurred")
finally:
    cleanup()

# EAFP vs LBYL
# EAFP (Easier to Ask for Forgiveness than Permission) - Pythonic
try:
    value = my_dict['key']
except KeyError:
    value = default

# LBYL (Look Before You Leap) - Less Pythonic
if 'key' in my_dict:
    value = my_dict['key']
else:
    value = default

# Context managers for resource handling
with open('file.txt', 'r') as f:
    content = f.read()
# File automatically closed even if exception occurs
```

### Code Organization

```python
# Function documentation
def binary_search(arr, target):
    """
    Perform binary search on a sorted array.

    Args:
        arr: Sorted list of comparable elements
        target: Element to search for

    Returns:
        Index of target if found, -1 otherwise

    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    # ... implementation

# Type hints (Python 3.5+)
from typing import List, Dict, Optional, Tuple, Union

def process_data(
    items: List[int],
    mapping: Dict[str, int]
) -> Optional[Tuple[int, str]]:
    # ... implementation
    pass

# Constants
HASH_TABLE_SIZE = 1009
MAX_ITERATIONS = 1000
DEFAULT_VALUE = -1

# Use enums for constants
from enum import Enum

class Direction(Enum):
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'
```

### Common Patterns

```python
# Iterate with index
for i, value in enumerate(items):
    print(f"Index {i}: {value}")

# Iterate over multiple sequences
for name, age, city in zip(names, ages, cities):
    print(f"{name} is {age} years old and lives in {city}")

# Dictionary iteration
for key, value in my_dict.items():
    print(f"{key}: {value}")

# Reverse iteration
for item in reversed(items):
    print(item)

# Conditional assignment
value = x if condition else y

# Multiple assignment
a, b = b, a  # swap
x, y, *rest = sequence  # unpack with remainder

# Default dictionary values
count = {}
for item in items:
    count[item] = count.get(item, 0) + 1

# Or use Counter
from collections import Counter
count = Counter(items)

# Flatten nested lists
nested = [[1, 2], [3, 4], [5, 6]]
flat = [item for sublist in nested for item in sublist]

# Group by key
from collections import defaultdict
groups = defaultdict(list)
for item in items:
    groups[get_key(item)].append(item)
```

---

### Interview-Specific Tips

1. **Always clarify the problem**:
   - Understand input/output format
   - Ask about edge cases
   - Confirm assumptions

2. **Start with brute force**:
   - Get a working solution first
   - Then optimize

3. **Think out loud**:
   - Explain your approach
   - Walk through examples
   - Discuss trade-offs

4. **Test your code**:
   - Walk through with examples
   - Consider edge cases
   - Check for off-by-one errors

5. **Know these patterns**:
   - Two pointers
   - Sliding window
   - Binary search
   - DFS/BFS
   - Dynamic programming
   - Backtracking

6. **Time/Space complexity**:
   - Always analyze and state complexity
   - Know Big O notation
   - Discuss trade-offs

Remember: Practice regularly on platforms like LeetCode, and focus on understanding patterns rather than memorizing solutions!

---

*Happy coding! 🚀*