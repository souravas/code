def move_zeros1(nums: list[int]) -> None:
    slow = 0

    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1

    for i in range(slow, len(nums)):
        nums[i] = 0


def move_zeros2(nums: list[int]) -> None:
    slow = 0

    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
