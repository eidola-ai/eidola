#!/usr/bin/env python3
"""Classify a byte difference between two Mach-Os as the known one, or not.

Task 55. `codesign` rounds `__LINKEDIT`'s `vmsize` up to 16 KiB on every
slice; signapple rounds it to the slice's own code-hash page size
(`sign.py:706`), and its `PAGE_SIZES` table is keyed by cputype: 4 KiB for
x86_64, 16 KiB for arm64/arm64e. So `signapple apply` reproduces a
`codesign`-signed universal binary exactly except, sometimes, for that one
field **on the x86_64 slice**. The divergence is documented in
work/reference/55-apple-signing/round-trip.md and fixed by one line in the
signapple fork.

This says which case a given diff is, so the round-trip harness can treat
the characterized divergence as a known state and anything else as a
genuine failure. A diff that has merely *grown* to include another field —
or another slice, or a value neither rounding rule explains — is the
failure this exists to catch, so being permissive here would defeat the
whole point. The known case is admitted only when all three hold:

  * every differing byte lies in an x86_64 slice's `vmsize` field;
  * the `codesign` side is `round_up(__LINKEDIT filesize, 16 KiB)`;
  * the signapple side is `round_up(__LINKEDIT filesize, 4 KiB)`.

The argument order is therefore load-bearing, and asymmetric.

Prints one of: identical | linkedit-vmsize-only | other
Exit status is 0 for the first two, 1 for `other`.

Usage: apple_linkedit_diff.py <codesign-signed> <apply-output>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macho_facts import facts  # noqa: E402  (sibling script, not a package)

# `codesign` on macOS 26 rounds every slice's __LINKEDIT vmsize to this,
# verified on a thin x86_64 binary so it is not a universal-binary quirk
# (round-trip.md §3.3).
CODESIGN_PAGE = 0x4000

# signapple's PAGE_SIZES, by arch. Only x86_64 disagrees with codesign.
SIGNAPPLE_PAGE = {"x86_64": 0x1000, "arm64": 0x4000, "arm64e": 0x4000}

DIVERGENT_ARCHS = tuple(a for a, p in SIGNAPPLE_PAGE.items() if p != CODESIGN_PAGE)


def round_up(value, page):
    return (value + page - 1) // page * page


def first_diff(a, b, lo, hi):
    for i in range(lo, hi):
        if a[i] != b[i]:
            return i
    return None


def classify(signed_path, applied_path):
    """Compare a `codesign`-signed Mach-O against `signapple apply`'s output."""
    with open(signed_path, "rb") as f:
        a = f.read()
    with open(applied_path, "rb") as f:
        b = f.read()
    if a == b:
        return "identical", []

    if len(a) != len(b):
        return "other", ["file lengths differ: %d vs %d" % (len(a), len(b))]

    # Only the x86_64 slices' vmsize fields may differ at all. Every other
    # byte, including the arm64 slices' vmsize, is graded as a failure.
    permitted = {}
    for sl in facts(signed_path)["slices"]:
        if sl["arch"] in DIVERGENT_ARCHS:
            permitted[sl["linkedit"]["vmsize_field_offset"]] = sl

    # Walk the gaps between permitted fields with slice comparisons, so the
    # common case never scans an 80 MB file byte by byte.
    bounds = sorted(permitted)
    edges = [0]
    for off in bounds:
        edges.extend((off, off + 8))
    edges.append(len(a))
    for lo, hi in zip(edges[0::2], edges[1::2]):
        if a[lo:hi] != b[lo:hi]:
            stray = first_diff(a, b, lo, hi)
            differing = sum(1 for i in range(len(a)) if a[i] != b[i])
            return "other", [
                "%d differing byte(s), first one outside an x86_64 "
                "__LINKEDIT vmsize at %#x" % (differing, stray)
            ]

    detail = []
    problems = []
    for off in bounds:
        va = int.from_bytes(a[off : off + 8], "little")
        vb = int.from_bytes(b[off : off + 8], "little")
        if va == vb:
            continue
        sl = permitted[off]
        filesize = sl["linkedit"]["filesize"]
        want_a = round_up(filesize, CODESIGN_PAGE)
        want_b = round_up(filesize, SIGNAPPLE_PAGE[sl["arch"]])
        detail.append(
            "%s slice: vmsize %#x vs %#x at file offset %#x "
            "(__LINKEDIT filesize %#x)" % (sl["arch"], va, vb, off, filesize)
        )
        if va != want_a:
            problems.append(
                "  codesign side %#x is not round_up(filesize, %#x) = %#x"
                % (va, CODESIGN_PAGE, want_a)
            )
        if vb != want_b:
            problems.append(
                "  apply side %#x is not round_up(filesize, %#x) = %#x"
                % (vb, SIGNAPPLE_PAGE[sl["arch"]], want_b)
            )

    if problems:
        return "other", detail + ["neither value matches the documented rule:"] + problems
    return "linkedit-vmsize-only", detail


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    kind, detail = classify(sys.argv[1], sys.argv[2])
    print(kind)
    for line in detail:
        print("  " + line)
    sys.exit(1 if kind == "other" else 0)
