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
record. It holds the exact unsigned regular-file set and hashes, plus the
output hashes per Mach-O, so applying to the wrong build is a refusal rather
than a corruption, and the signed artifact's
**per-slice structural facts** — the fat-header placement, `__LINKEDIT`'s
sizing, and where each superblob lands. Those facts are what let `apply`
reproduce the signed bundle from the artifact *as built*, without settling it
first and without reimplementing `codesign`'s arithmetic: it writes the
recorded values rather than deriving them. `scripts/apple-place.py` is the
apply side, and the round-trip harness grades it.

Usage: apple-detach.py <signed bundle> <output dir> <unsigned bundle>
"""

import hashlib
import json
import os
import shutil
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macho_facts import facts  # noqa: E402  (sibling script, not a package)

# signapple names detached files by cputype alone (sign.py CPU_NAMES).
ARCH_SUFFIX = {"arm64": "arm64", "arm64e": "arm64", "x86_64": "x86_64"}
RECORD_NAME = "eidola-placement.json"


def is_within(path, root):
    """Return whether canonical `path` is equal to or below canonical `root`."""
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def validate_destination(bundle, unsigned, out_dir, bundle_name):
    """Refuse current output paths that overlap either source bundle."""
    signed_source = os.path.realpath(bundle)
    unsigned_source = os.path.realpath(unsigned)
    output = os.path.realpath(os.path.abspath(out_dir))
    material = os.path.join(output, bundle_name)
    for source in (signed_source, unsigned_source):
        if (
            is_within(output, source)
            or is_within(material, source)
            or is_within(source, material)
        ):
            raise SystemExit(f"detached output overlaps source: {source}")


def validate_existing_output(out_dir):
    """Accept only an empty root or one complete parseable prior output."""
    if not os.path.lexists(out_dir):
        return None
    mode = os.lstat(out_dir).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise SystemExit(f"detached output is not a plain directory: {out_dir}")

    names = sorted(os.listdir(out_dir))
    if not names:
        return None
    if RECORD_NAME not in names:
        raise SystemExit(f"unexpected detached output entry: {names[0]}")

    record = os.path.join(out_dir, RECORD_NAME)
    if not stat.S_ISREG(os.lstat(record).st_mode):
        raise SystemExit(f"invalid previous placement record: {record}")
    try:
        with open(record) as file:
            previous = json.load(file).get("bundle")
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid previous placement record: {record}") from error
    if (
        not isinstance(previous, str)
        or not previous
        or os.path.basename(previous) != previous
    ):
        raise SystemExit(f"invalid previous placement record: {record}")

    expected = {RECORD_NAME, previous}
    unexpected = sorted(set(names) - expected)
    if unexpected:
        raise SystemExit(f"unexpected detached output entry: {unexpected[0]}")
    missing = sorted(expected - set(names))
    if missing:
        raise SystemExit(f"previous detached output is missing: {missing[0]}")
    previous_root = os.path.join(out_dir, previous)
    previous_mode = os.lstat(previous_root).st_mode
    if stat.S_ISLNK(previous_mode) or not stat.S_ISDIR(previous_mode):
        raise SystemExit(f"previous detached app is not a plain directory: {previous}")
    return previous


def validate_previous_destination(bundle, unsigned, out_dir, previous):
    """Refuse a validated previous cleanup root that overlaps a source."""
    if previous is None:
        return
    previous_root = os.path.realpath(os.path.join(out_dir, previous))
    for source in (os.path.realpath(bundle), os.path.realpath(unsigned)):
        if is_within(previous_root, source) or is_within(source, previous_root):
            raise SystemExit(f"previous detached root overlaps source: {source}")


def clear_previous(out_dir, root, previous):
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
    record = os.path.join(out_dir, RECORD_NAME)
    if previous is not None:
        previous_root = os.path.join(out_dir, previous)
        shutil.rmtree(previous_root)
        os.remove(record)
    if os.path.isdir(root):
        shutil.rmtree(root)


def unsigned_inputs(unsigned):
    """Return the normalized regular-file tree without following symlinks."""
    inputs = {}
    for directory, dirs, files in os.walk(unsigned):
        for name in dirs + files:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                raise SystemExit(f"unsigned input is a symbolic link: {path}")
        for name in files:
            path = os.path.join(directory, name)
            rel = os.path.relpath(path, unsigned).replace(os.sep, "/")
            with open(path, "rb") as f:
                inputs[rel] = f"sha256:{hashlib.sha256(f.read()).hexdigest()}"
    return dict(sorted(inputs.items()))


def detach(bundle, out_dir, unsigned):
    bundle_name = os.path.basename(bundle.rstrip("/"))
    root = os.path.join(out_dir, bundle_name)
    validate_destination(bundle, unsigned, out_dir, bundle_name)
    previous = validate_existing_output(out_dir)
    validate_previous_destination(bundle, unsigned, out_dir, previous)
    os.makedirs(out_dir, exist_ok=True)
    clear_previous(out_dir, root, previous)
    macos_out = os.path.join(root, "Contents", "MacOS")
    os.makedirs(macos_out)

    macos_dir = os.path.join(bundle, "Contents", "MacOS")
    placement = {
        "schema_version": 1,
        "bundle": bundle_name,
        "inputs": unsigned_inputs(unsigned),
        "machos": {},
        "files": {},
    }

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
            # The structural facts `apply` writes onto the unsigned build,
            # verbatim as macho_facts.py reports them for the signed one:
            # fat offset/size/align, __LINKEDIT's sizing and the file offset
            # of each of its fields, and the superblob's placement and hash.
            "kind": info["kind"],
            "slices": info["slices"],
            "output_sha256": info["file_sha256"],
            "output_len": info["file_size"],
        }
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

    with open(os.path.join(out_dir, RECORD_NAME), "w") as f:
        json.dump(placement, f, indent=2, sort_keys=True)
        f.write("\n")

    return root


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    print(detach(sys.argv[1], sys.argv[2], sys.argv[3]))
