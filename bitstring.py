import sys
from typing import List

def load(filename: str) -> bytes:
    with open(filename, 'rb') as file:
        return file.read()

def main(args: List[str]) -> int:
    print(args)
    a = load(args[1])
    with open(args[2], 'w') as outfile:
        bits = 0
        for b in a:
            for _ in range(8):
                print(b & 1, end=('\n' if bits & 127 == 127 else ''), file=outfile)
                b >>= 1
                bits += 1

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
