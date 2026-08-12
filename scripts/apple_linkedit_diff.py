#!/usr/bin/env python3
"""Classify a byte difference between two Mach-Os as the known one, or not.

Task 55. `codesign` rounds `__LINKEDIT`'s `vmsize` up to 16 KiB on every
slice; signapple rounds it to the slice's own code-hash page size, which is
4 KiB on x86_64 (`sign.py:706`). So `signapple apply` reproduces a
`codesign`-signed universal binary exactly except, sometimes, for that one
field on the x86_64 slice. The divergence is documented in
work/reference/55-apple-signing/round-trip.md and fixed by one line in the
signapple fork.

This says which case a given diff is, so the round-trip harness can treat
the characterized divergence as a known state and anything else as a
genuine failure. A diff that has merely *grown* to include another field is
the failure this exists to catch.

Prints one of: identical | linkedit-vmsize-only | other
Exit status is 0 for the first two, 1 for `other`.

Usage: apple_linkedit_diff.py <a> <b>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macho_facts import facts  # noqa: E402  (sibling script, not a package)


def classify(path_a, path_b):
    with open(path_a, "rb") as f:
        a = f.read()
    with open(path_b, "rb") as f:
        b = f.read()
    if a == b:
        return "identical", []

    allowed = set()
    for sl in facts(path_a)["slices"]:
        start = sl["linkedit"]["vmsize_field_offset"]
        allowed.update(range(start, start + 8))

    if len(a) != len(b):
        return "other", ["file lengths differ: %d vs %d" % (len(a), len(b))]

    differing = [i for i in range(len(a)) if a[i] != b[i]]
    stray = [i for i in differing if i not in allowed]
    if stray:
        return "other", ["%d differing byte(s), %d outside __LINKEDIT vmsize, first at %#x"
                         % (len(differing), len(stray), stray[0])]

    detail = []
    for sl in facts(path_a)["slices"]:
        off = sl["linkedit"]["vmsize_field_offset"]
        if any(off <= i < off + 8 for i in differing):
            va = int.from_bytes(a[off:off + 8], "little")
            vb = int.from_bytes(b[off:off + 8], "little")
            detail.append("%s slice: vmsize %#x vs %#x at file offset %#x"
                          % (sl["arch"], va, vb, off))
    return "linkedit-vmsize-only", detail


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    kind, detail = classify(sys.argv[1], sys.argv[2])
    print(kind)
    for line in detail:
        print("  " + line)
    sys.exit(1 if kind == "other" else 0)
