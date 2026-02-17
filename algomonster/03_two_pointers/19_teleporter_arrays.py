MODULO_AMT = 10**9 + 7


def maximum_score(arr1: list[int], arr2: list[int]) -> int:
    result = 0
    ptr1 = 0
    ptr2 = 0
    n1 = len(arr1)
    n2 = len(arr2)

    section_sum1 = 0
    section_sum2 = 0

    while ptr1 < n1 or ptr2 < n2:
        if ptr1 < n1 and ptr2 < n2 and arr1[ptr1] == arr2[ptr2]:
            result += max(section_sum1, section_sum2) + arr1[ptr1]
            result %= MODULO_AMT
            section_sum1 = 0
            section_sum2 = 0
            ptr1 += 1
            ptr2 += 1
            continue
        if ptr1 < n1 and ptr2 < n2:
            if arr1[ptr1] < arr2[ptr2]:
                section_sum1 += arr1[ptr1]
                ptr1 += 1
            else:
                section_sum2 += arr2[ptr2]
                ptr2 += 1
        else:
            if ptr1 == n1:
                section_sum2 += arr2[ptr2]
                ptr2 += 1
            else:
                section_sum1 += arr1[ptr1]
                ptr1 += 1
    result += max(section_sum1, section_sum2)
    return result % MODULO_AMT
