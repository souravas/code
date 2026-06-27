def maximum_score(arr1: list[int], arr2: list[int]) -> int:
    modulo_amount = 10**9 + 7
    result = 0
    index1 = 0
    index2 = 0

    n1 = len(arr1)
    n2 = len(arr2)

    section_sum1 = 0
    section_sum2 = 0


    while index1 < n1 and index2 < n2:
        if arr1[index1] == arr2[index2]:
            result += max(section_sum1, section_sum2) + arr1[index1]
            result %= modulo_amount
            section_sum1 = 0
            section_sum2 = 0
            index1 += 1
            index2 += 1
        elif arr1[index1] < arr2[index2]:
            section_sum1 += arr1[index1]
            index1 += 1
        else:
            section_sum2 += arr2[index2]
            index2 += 1
    while index1 < n1:
        section_sum1 += arr1[index1]
        index1 += 1
    while index2 < n2:
        section_sum2 += arr2[index2]
        index2 += 1
    result += max(section_sum1, section_sum2)
    return result % modulo_amount
