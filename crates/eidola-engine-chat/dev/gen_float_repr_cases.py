#!/usr/bin/env python3
"""Generate tests/fixtures/float_repr_cases.json: binary64 values and the text
CPython's `float.__repr__` (which `json.dumps` uses) prints for each.

The cases favour where shortest-digit algorithms disagree: exact ties between
two shortest candidates (decimals with 16-17 significant digits whose binary
value ends in ...5), powers of two (asymmetric rounding intervals), the
fixed/exponent switch at 1e16 and 1e-4, subnormals and the extremes, plus
uniformly random bit patterns and random decimals over every magnitude.

Dev-only; the output is committed. Deterministic (fixed seed):

    python -I dev/gen_float_repr_cases.py
"""

import json
import math
import pathlib
import random
import struct

CRATE = pathlib.Path(__file__).resolve().parent.parent
OUT = CRATE / "tests" / "fixtures" / "float_repr_cases.json"


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def from_bits(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def main():
    rng = random.Random(0x5EED_F10A7)
    values = set()

    def add(x):
        if math.isfinite(x):
            values.add(x)
            values.add(-x)

    # Exact ties: k + j/8 with ~15-17 significant digits, the reported shape.
    for _ in range(1500):
        k = rng.randrange(10**13, 10**15)
        add(k + rng.choice([0.125, 0.375, 0.625, 0.875, 0.25, 0.75, 0.5]))
    for _ in range(800):
        e = rng.randrange(-60, 60)
        m = rng.randrange(2**52, 2**53)
        add(math.ldexp(m, e))
    # Powers of two and their neighbours.
    for e in range(-1074, 1024):
        p = math.ldexp(1.0, e)
        add(p)
        if e % 4 == 0:
            add(math.nextafter(p, 0.0))
    # Around the fixed/exponent switch and powers of ten.
    for e in range(-30, 30):
        p = float(f"1e{e}")
        for x in (p, math.nextafter(p, math.inf), math.nextafter(p, 0.0), 9.999999999999998 * p):
            add(x)
    # Random bit patterns over every exponent, subnormals included.
    for _ in range(2500):
        add(from_bits(rng.getrandbits(63)))
    # Random short and long decimals over many magnitudes.
    for _ in range(2000):
        digits = rng.randrange(1, 18)
        mant = rng.randrange(10 ** (digits - 1), 10**digits)
        add(float(f"{mant}e{rng.randrange(-330, 300)}"))
    for x in (181703637716804.12, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 0.1, 0.3, 1 / 3, 2 / 3):
        add(x)

    cases = sorted(([f"{bits(x):016x}", repr(x)] for x in values), key=lambda c: c[0])
    OUT.write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
