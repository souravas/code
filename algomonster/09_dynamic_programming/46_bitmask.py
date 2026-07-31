# Binary Numbers: Quick Review
# Computers store integers in binary. Each position represents a power of 2:

# Binary:  1  0  1  0  1
#          ↓  ↓  ↓  ↓  ↓
# Power:  2⁴ 2³ 2² 2¹ 2⁰
# Value:  16  0  4  0  1  = 21
# Python shows binary with the bin() function:

# >>> bin(21)
# '0b10101'
# The 0b prefix indicates binary. Leading zeros are omitted.

# Bitwise Operations
# These operations work on individual bits of integers.

# AND (&): Both bits must be 1
#   0 1 0 1 0 1  (21)
# & 1 0 0 1 0 1  (37)
# =============
#   0 0 0 1 0 1  (5)
# Use case: Check if a bit is set. mask & (1 << i) is non-zero if bit i is 1.

# OR (|): Either bit is 1
#   0 1 0 1 0 1  (21)
# | 1 0 0 1 0 1  (37)
# =============
#   1 1 0 1 0 1  (53)
# Use case: Set a bit. mask | (1 << i) sets bit i to 1.

# XOR (^): Bits are different
#   0 1 0 1 0 1  (21)
# ^ 1 0 0 1 0 1  (37)
# =============
#   1 1 0 0 0 0  (48)
# Use case: Toggle a bit. mask ^ (1 << i) flips bit i.

# Left Shift (<<): Multiply by 2
# 21 << 2:
#   0 0 1 0 1 0 1  →  1 0 1 0 1 0 0  = 84
# x << n multiplies x by 2ⁿ. Creating a mask for bit i: 1 << i.

# Right Shift (>>): Divide by 2
# 21 >> 2:
#   0 1 0 1 0 1  →  0 0 0 1 0 1  = 5
# x >> n divides x by 2ⁿ (floor division). Bits shifted past the right edge disappear.

# Common Bitmask Operations
# Check if bit i is set
# if mask & (1 << i):  # non-zero means bit i is 1
#     print(f"Element {i} is in the set")
# Set bit i (add element to set)
# new_mask = mask | (1 << i)
# Clear bit i (remove element from set)
# new_mask = mask & ~(1 << i)
# Toggle bit i
# new_mask = mask ^ (1 << i)
# Count set bits (set size)
# count = bin(mask).count('1')  # or use popcount in other languages


# Generating All Subsets
# For an array of n elements, there are 2ⁿ possible subsets. Each subset corresponds to a bitmask from 0 to 2ⁿ - 1.

# nums = [1, 2, 3]
# n = len(nums)

# for mask in range(1 << n):  # 0 to 7 for n=3
#     subset = []
#     for i in range(n):
#         if mask & (1 << i):
#             subset.append(nums[i])
#     print(f"mask={mask:03b}: {subset}")
# Output:

# mask=000: []
# mask=001: [1]
# mask=010: [2]
# mask=011: [1, 2]
# mask=100: [3]
# mask=101: [1, 3]
# mask=110: [2, 3]
# mask=111: [1, 2, 3]