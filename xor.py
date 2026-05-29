import sys
from typing import List

# FIXME: we still fail near the very tail of larger files, like the compression.exe binaries

def load(filename: str) -> bytes:
    with open(filename, 'rb') as file:
        return file.read()

def main(args: List[str]) -> int:
    print(args)
    a = load(args[1])
    b = load(args[2])
    if len(a) != len(b):
        print(f"Lengths differ: (len(a) = {len(a)}, len(b) = {len(b)})")

    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            print("First difference at offset", i)
            break

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
