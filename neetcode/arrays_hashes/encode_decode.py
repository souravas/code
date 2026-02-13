class Solution:

    def encode(self, strs: list[str]) -> str:
        encoded_string = ""
        for string in strs:
            encoded_string += f"{len(string)}#{string}"
        return encoded_string

    def decode(self, s: str) -> list[str]:
        result = []
        i = 0
        while i < len(s):
            j = i
            while s[j] != '#':
                j += 1
            length = int(s[i:j])
            i = j + 1
            j = i + length
            result.append(s[i:j])
            i = j
        return result
