def least_consecutive_cards_to_match(cards: list[int]) -> int:
    seen = set()
    left = 0
    least = len(cards) + 1

    for right in range(len(cards)):
        while cards[right] in seen:
            least = min(least, right - left + 1)
            seen.remove(cards[left])
            left += 1
        seen.add(cards[right])

    if least > len(cards):
        return -1

    return least
