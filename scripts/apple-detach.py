#!/usr/bin/env python3
"""Detach a bundle's code signatures into signapple's on-disk layout.

signapple has no keyless detach: `signapple sign --detach` is the only path
that emits a detached bundle, and it takes a PKCS#12 archive — a
non-exportable Developer ID key on a hardware token can never feed it. So
the signing side detaches from a `codesign`-produced bundle instead, and
signapple's `apply` stays the independent implementation that checks us.

The format is signapple's, verbatim, so `signapple apply` consumes this
output unchanged:

    <out>/<Bundle>.app/Contents/MacOS/<name>.<arch>sign   per slice
    <out>/<Bundle>.app/Contents/_CodeSignature/CodeResources

A detached signature file is exactly the bytes `LC_CODE_SIGNATURE` points
at — `[dataoff, dataoff + datasize)` within the slice — which is what
signapple writes and what it reads back.

Plus one file signapple ignores, `eidola-placement.json`, the placement
record: it holds the input and output hashes per Mach-O, so applying to the
wrong build is a refusal rather than a corruption.

Usage: apple-detach.py <signed bundle> <output dir> [unsigned bundle]

The optional third argument is the unsigned build the signature was taken
from; giving it records `input_sha256` per Mach-O in the placement record.
"""

import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macho_facts import facts  # noqa: E402  (sibling script, not a package)

# signapple names detached files by cputype alone (sign.py CPU_NAMES).
ARCH_SUFFIX = {"arm64": "arm64", "arm64e": "arm64", "x86_64": "x86_64"}


def clear_previous(out_dir, root):
    """Remove what a previous detach into `out_dir` left behind.

    Regenerating into a reused directory must not keep material the new
    input no longer has. If a later build drops a slice, an executable, the
    bundle seal or a stapled ticket, the old `.archsign` or `CodeResources`
    would survive and `signapple apply` would consume stale material that no
    placement record mentions. The documented fixture-regeneration command
    reuses `detached/`, so this is the ordinary path, not the exotic one.

    Only what this script writes is removed: the bundle tree it is about to
    write, and the one a previous placement record names.
    """
    record = os.path.join(out_dir, "eidola-placement.json")
    if os.path.isfile(record):
        try:
            with open(record) as f:
                previous = json.load(f).get("bundle")
        except (OSError, ValueError):
            previous = None
        # A basename and nothing else, so a damaged record cannot point the
        # removal outside `out_dir`.
        if isinstance(previous, str) and previous and os.path.basename(previous) == previous:
            previous_root = os.path.join(out_dir, previous)
            if os.path.isdir(previous_root):
                shutil.rmtree(previous_root)
        os.remove(record)
    if os.path.isdir(root):
        shutil.rmtree(root)


def detach(bundle, out_dir, unsigned=None):
    bundle_name = os.path.basename(bundle.rstrip("/"))
    root = os.path.join(out_dir, bundle_name)
    os.makedirs(out_dir, exist_ok=True)
    clear_previous(out_dir, root)
    macos_out = os.path.join(root, "Contents", "MacOS")
    os.makedirs(macos_out)

    macos_dir = os.path.join(bundle, "Contents", "MacOS")
    placement = {"schema_version": 1, "bundle": bundle_name, "machos": {}, "files": {}}

    for name in sorted(os.listdir(macos_dir)):
        path = os.path.join(macos_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            data = f.read()
        info = facts(path)
        rel = os.path.join("Contents", "MacOS", name)
        for sl in info["slices"]:
            cs = sl["code_signature"]
            if cs is None:
                raise SystemExit(f"{rel} slice {sl['arch']} carries no signature")
            base = sl.get("fat_offset", 0)
            blob = data[base + cs["dataoff"] : base + cs["dataoff"] + cs["datasize"]]
            suffix = ARCH_SUFFIX[sl["arch"]]
            with open(os.path.join(macos_out, f"{name}.{suffix}sign"), "wb") as f:
                f.write(blob)
        record = {
            "output_sha256": info["file_sha256"],
            "output_len": info["file_size"],
        }
        if unsigned is not None:
            with open(os.path.join(unsigned, rel), "rb") as f:
                record["input_sha256"] = hashlib.sha256(f.read()).hexdigest()
        placement["machos"][rel] = record

    seal = os.path.join(bundle, "Contents", "_CodeSignature", "CodeResources")
    if os.path.isfile(seal):
        dest_dir = os.path.join(root, "Contents", "_CodeSignature")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copyfile(seal, os.path.join(dest_dir, "CodeResources"))
        with open(seal, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        placement["files"]["Contents/_CodeSignature/CodeResources"] = f"sha256:{digest}"

    # The notarization ticket, when one has been stapled. Outside the
    # code-signature seal, so it travels as a plain file.
    ticket = os.path.join(bundle, "Contents", "CodeResources")
    if os.path.isfile(ticket):
        shutil.copyfile(ticket, os.path.join(root, "Contents", "CodeResources"))
        with open(ticket, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        placement["files"]["Contents/CodeResources"] = f"sha256:{digest}"

    with open(os.path.join(out_dir, "eidola-placement.json"), "w") as f:
        json.dump(placement, f, indent=2, sort_keys=True)
        f.write("\n")

    return root


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    print(detach(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None))
