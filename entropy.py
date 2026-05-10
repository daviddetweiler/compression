"""Experiments in computing Shannon entropy"""

import sys
from math import log2
from typing import Type


class Model:
    """Base class for bytewise probability models"""

    def pvalue(self, v: int) -> float:
        """Probability that v is the next byte"""
        _ = v
        return 1 / 256

    def update(self, v: int) -> None:
        """Update model based on actual next byte"""
        _ = v
        return

    def entropy(self) -> float:
        """Compute current entropy per byte of model"""
        e = 0.0
        for b in range(256):
            p = self.pvalue(b)
            e -= p * log2(p)

        return e


class UniformModel(Model):
    """Always assumes every byte is equally likely"""


class SimpleByteModel(Model):
    """Keeps track of past byte frequencies"""

    def __init__(self):
        self.histogram = [1] * 256
        self.total = 256

    def pvalue(self, v: int) -> float:
        count = self.histogram[v]
        total = self.total
        return count / total

    def update(self, v: int) -> None:
        self.histogram[v] += 1
        self.total += 1


class SimpleMarkovModel(Model):
    """Uses the previous byte to help predict the current one"""

    def __init__(self):
        self.histogram = [[1] * 256 for _ in range(256)]
        self.total = [256] * 256
        self.last = 0

    def pvalue(self, v: int) -> float:
        count = self.histogram[self.last][v]
        total = self.total[self.last]
        return count / total

    def update(self, v: int) -> None:
        self.histogram[self.last][v] += 1
        self.total[self.last] += 1
        self.last = v


def entropy(content: bytes, model_type: Type[Model]) -> float:
    """Assuming iid. bytes, computes byte-wise entropy"""
    e = 0.0
    model = model_type()
    for b in content:
        model.update(b)
        e += model.entropy()

    return e


def popcnt(b: int) -> int:
    """Number of set bits in a byte"""
    assert b >= 0 and b < 256
    n = 0
    for _ in range(8):
        bit = b & 1
        if bit:
            n += 1

    return n


POPCNT_LUT = [popcnt(b) for b in range(256)]


def bitwise_entropy(data: bytes) -> float:
    """Assuming iid. bits in the datastream, computes the bitwise entropy"""
    total = 8 * len(data)
    ones = 0
    for b in data:
        ones += POPCNT_LUT[b]

    return sum(-p * log2(p) for p in [ones / total, (total - ones) / total])


def main():
    """Entrypoint"""
    filename = sys.argv[1]
    with open(filename, "rb") as file:
        data = file.read()

    l = len(data)
    print("Actual size:", l)
    histogram = [0] * 256
    total = 0
    for b in data:
        histogram[b] += 1
        total += 1

    e = 0.0
    for f in histogram:
        p = f / total
        e -= (p * log2(p)) if p > 0.0 else 0.0

    print("Actual total entropy content:", e * l / 8)

    print("Bitwise entropy:", bitwise_entropy(data))

    for mtype in [UniformModel, SimpleByteModel, SimpleMarkovModel]:
        print(f"{mtype.__name__}: {entropy(data, mtype) / 8}")


if __name__ == "__main__":
    main()
