#!/usr/bin/env python3
"""Reattach a detached signature using the placement record's structural facts.

This is the apply side of the design the round trip settled on: the build
does not settle, so `apply` receives the artifact **as built** — carrying
sigtool's linker-signed signatures, its own fat alignment and its own
`__LINKEDIT` sizing — and has to land on `codesign`'s layout anyway.

It does that without knowing any of `codesign`'s arithmetic. `detach` walked
the signed bundle, so `eidola-placement.json` already names the target
layout; this writes what the record says:

  * the fat header, rebuilt at the recorded per-slice offset/size/align
    (the cputype/cpusubtype come from the input, which signing never moves);
  * `__LINKEDIT`'s `vmsize`/`fileoff`/`filesize`, at the recorded field
    offsets — so the 16 KiB-vs-4 KiB rounding question never arises here,
    because nothing is rounded;
  * `LC_CODE_SIGNATURE`'s `dataoff`/`datasize`, and the superblob at that
    offset;
  * the mach header's flags.

Everything before the signature is copied from the input slice unchanged,
which is the assumption the whole approach rests on and which is checked, not
assumed: the input must supply every byte below the recorded `dataoff`, and
the finished file must match the recorded `output_sha256`. A future
`codesign` that rewrote something else in the slice would fail that hash
rather than ship.

A measurement harness, not the shipping implementation — `eidola-apple::apply`
is what ships. This exists so the design's central equation is demonstrated on
a real artifact rather than argued:

    place(as-built, detached) == the codesign-signed bundle, byte for byte

Applies in place, like `signapple apply`, to a bundle that is already a
writable copy of the unsigned build.

Usage: apple-place.py <bundle> <detached bundle dir>

The second argument is the directory `apple-detach.py` wrote, i.e. the one
holding `eidola-placement.json`, or the `<Bundle>.app` inside it — either is
accepted, since `signapple apply` takes the latter.
"""

import hashlib
import json
import os
import shutil
import stat
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macho_facts import (  # noqa: E402  (sibling script, not a package)
    FAT_ARCH,
    FAT_ARCH_64,
    FAT_MAGIC_64,
    cpu_name,
    facts,
)

# signapple names detached files by cputype alone (sign.py CPU_NAMES).
ARCH_SUFFIX = {"arm64": "arm64", "arm64e": "arm64", "x86_64": "x86_64"}
RECORD_NAME = "eidola-placement.json"

# Offset of `flags` in a 64-bit mach header, and of `dataoff` in a
# linkedit_data_command (past cmd/cmdsize).
MH_FLAGS_OFFSET = 24
LC_DATAOFF_OFFSET = 8


def validate_detached_material(root, record):
    """Require exactly the signature blobs and plain files the record names."""
    expected = set(record["files"])
    for rel, entry in record["machos"].items():
        name = os.path.basename(rel)
        for target in entry["slices"]:
            expected.add(
                f"Contents/MacOS/{name}.{ARCH_SUFFIX[target['arch']]}sign"
            )

    actual = set()
    for directory, dirs, files in os.walk(root):
        for name in dirs + files:
            path = os.path.join(directory, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                raise SystemExit(f"{rel}: detached material is a symbolic link")
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise SystemExit(f"{rel}: detached material is not a regular file")
        for name in files:
            path = os.path.join(directory, name)
            actual.add(os.path.relpath(path, root).replace(os.sep, "/"))
    unexpected = sorted(actual - expected)
    if unexpected:
        raise SystemExit(f"{unexpected[0]}: unexpected detached material")


def require_plain_directory(path, label):
    """Refuse roots that are symbolic links or non-directories."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as error:
        raise SystemExit(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(mode):
        raise SystemExit(f"{label} is a symbolic link: {path}")
    if not stat.S_ISDIR(mode):
        raise SystemExit(f"{label} is not a directory: {path}")


def is_plain_file(path):
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def validate_detached_root(detached, record):
    """Require one placement record plus exactly the recorded app tree."""
    require_plain_directory(detached, "detached archive root")
    for name in sorted(os.listdir(detached)):
        path = os.path.join(detached, name)
        mode = os.lstat(path).st_mode
        if name == RECORD_NAME:
            if stat.S_ISREG(mode):
                continue
            raise SystemExit(f"{name}: placement record is not a regular file")
        if name == record["bundle"]:
            if stat.S_ISDIR(mode):
                continue
            raise SystemExit(f"{name}: recorded app tree is not a directory")
        if stat.S_ISREG(mode):
            raise SystemExit(f"{name}: unexpected detached material")
        if stat.S_ISLNK(mode):
            raise SystemExit(f"{name}: detached archive root contains a symbolic link")
        if stat.S_ISDIR(mode):
            raise SystemExit(f"{name}: detached archive root contains an unexpected directory")
        raise SystemExit(f"{name}: detached archive root contains a special file")


def rebuild(source, source_slices, record, signatures):
    """The signed Mach-O, rebuilt from the unsigned one plus the record."""
    by_arch = {sl["arch"]: sl for sl in source_slices}
    if record["kind"] == "fat":
        (magic,) = struct.unpack_from(">I", source, 0)
        (count,) = struct.unpack_from(">I", source, 4)
        entry = FAT_ARCH_64 if magic == FAT_MAGIC_64 else FAT_ARCH
        cpu = {}
        for i in range(count):
            fields = entry.unpack_from(source, 8 + i * entry.size)
            cpu[cpu_name(fields[0], fields[1])] = fields[:2]
        if count != len(source_slices):
            raise SystemExit("input fat table and parsed slices disagree")
        source_end = 8 + count * entry.size
        input_max_align = 0
        for source_slice in source_slices:
            arch = source_slice["arch"]
            if source_slice.get("fat_offset") != source_slice["header_offset"]:
                raise SystemExit(
                    f"{arch}: input fat offset and Mach header offset disagree"
                )
            align = source_slice.get("fat_align")
            if not isinstance(align, int) or align < 0 or align >= sys.maxsize.bit_length():
                raise SystemExit(f"{arch}: input fat alignment exponent is too large")
            input_max_align = max(input_max_align, align)
            alignment = 1 << align
            source_end = (source_end + alignment - 1) & ~(alignment - 1)
            if source_end > sys.maxsize:
                raise SystemExit(f"{arch}: input fat slice alignment overflow")
            if source_slice["fat_offset"] != source_end:
                raise SystemExit(f"{arch}: input fat table is not canonically packed")
            source_end += source_slice["fat_size"]
        if source_end != len(source):
            raise SystemExit(
                f"input fat slices end at {source_end}, file length is {len(source)}"
            )
    planned = []
    ranges = []

    for target in record["slices"]:
        arch = target["arch"]
        if arch not in by_arch:
            raise SystemExit(f"input has no {arch} slice for the record to place")
        cs = target["code_signature"]
        blob = signatures[arch]
        if len(blob) != cs["datasize"]:
            raise SystemExit(
                f"{arch}: detached signature is {len(blob)} bytes, "
                f"record says {cs['datasize']}"
            )
        if hashlib.sha256(blob).hexdigest() != cs["superblob_sha256"]:
            raise SystemExit(f"{arch}: detached signature is not the recorded superblob")

        base = target["header_offset"]
        if base < 0:
            raise SystemExit(f"{arch}: negative slice offset")
        source_base = by_arch[arch]["header_offset"]

        # The precondition the whole approach rests on: signing rewrites load
        # commands, it never adds or moves one, so the recorded field offsets
        # address the input as well as the output. That holds when the input
        # is already ad-hoc signed per slice, which is what the Nix build's
        # autoSignDarwinBinariesHook guarantees. A never-signed slice has no
        # LC_CODE_SIGNATURE to rewrite — `unsettled.macho`'s x86_64 slice in
        # the committed fixtures is one — and placing onto it would mean
        # *inserting* a load command and relaying everything after it. Out of
        # scope here, and said so rather than discovered as a hash mismatch.
        source_cs = by_arch[arch]["code_signature"]
        if source_cs is None:
            raise SystemExit(
                f"{arch}: input slice carries no LC_CODE_SIGNATURE; placement "
                "rewrites that load command, it does not insert one"
            )
        if (
            source_cs["dataoff"] != cs["dataoff"]
            or source_cs["lc_offset"] - source_base != cs["lc_offset"] - base
        ):
            raise SystemExit(
                f"{arch}: signing moved the load-command layout "
                f"(LC_CODE_SIGNATURE at {source_cs['lc_offset'] - source_base:#x} "
                f"-> {cs['lc_offset'] - base:#x}, signature at "
                f"{source_cs['dataoff']:#x} -> {cs['dataoff']:#x}); the recorded "
                "field offsets do not address this input"
            )

        head = bytearray(source[source_base : source_base + cs["dataoff"]])
        if len(head) != cs["dataoff"]:
            raise SystemExit(
                f"{arch}: input slice ends at {len(head)}, before the recorded "
                f"signature offset {cs['dataoff']}"
            )

        linkedit = target["linkedit"]
        for field in ("vmsize", "fileoff", "filesize"):
            struct.pack_into(
                "<Q", head, linkedit[f"{field}_field_offset"] - base, linkedit[field]
            )
        struct.pack_into(
            "<II",
            head,
            cs["lc_offset"] - base + LC_DATAOFF_OFFSET,
            cs["dataoff"],
            cs["datasize"],
        )
        struct.pack_into("<I", head, MH_FLAGS_OFFSET, int(target["mh_flags"], 16))

        body = bytes(head) + blob
        end = base + len(body)
        if any(base < prior_end and prior_start < end for prior_start, prior_end in ranges):
            raise SystemExit(f"{arch}: slice overlaps another recorded slice")
        if record["kind"] == "fat" and target.get("fat_size") != len(body):
            raise SystemExit(
                f"{arch}: slice is {len(body)} bytes, record says {target.get('fat_size')}"
            )
        planned.append((base, end, body))
        ranges.append((base, end))

    if record["kind"] == "thin":
        if len(ranges) != 1 or ranges[0][0] != 0:
            raise SystemExit("thin output is not exactly one slice at file offset zero")
        derived_len = ranges[0][1]
    else:
        derived_len = 8 + len(record["slices"]) * entry.size
        for target, (start, end), source_slice in zip(
            record["slices"], ranges, source_slices
        ):
            arch = target["arch"]
            if arch != source_slice["arch"]:
                raise SystemExit(f"{arch}: slice order disagrees with the input fat table")
            if target.get("fat_offset") != target["header_offset"]:
                raise SystemExit(f"{arch}: fat offset and Mach header offset disagree")
            align = target.get("fat_align")
            if not isinstance(align, int) or align < 0:
                raise SystemExit(f"{arch}: fat alignment exponent is too large")
            if align > input_max_align:
                raise SystemExit(
                    f"{arch}: fat alignment exponent {align} exceeds input maximum "
                    f"{input_max_align}"
                )
            if align >= sys.maxsize.bit_length():
                raise SystemExit(f"{arch}: fat alignment exponent is too large")
            alignment = 1 << align
            derived_len = (derived_len + alignment - 1) & ~(alignment - 1)
            if derived_len > sys.maxsize:
                raise SystemExit(f"{arch}: fat slice alignment overflow")
            if start != derived_len:
                raise SystemExit(
                    f"{arch}: fat slice starts at {start}, canonical packing "
                    f"requires {derived_len}"
                )
            derived_len = end
    if record["output_len"] != derived_len:
        raise SystemExit(
            f"recorded output length {record['output_len']} does not equal "
            f"reconstructed end {derived_len}"
        )

    out = bytearray(derived_len)
    for base, end, body in planned:
        out[base:end] = body

    if record["kind"] == "fat":
        struct.pack_into(">II", out, 0, magic, len(record["slices"]))
        for i, target in enumerate(record["slices"]):
            values = cpu[target["arch"]] + (
                target["fat_offset"],
                target["fat_size"],
                target["fat_align"],
            )
            if magic == FAT_MAGIC_64:
                values += (0,)  # reserved
            entry.pack_into(out, 8 + i * entry.size, *values)

    return bytes(out)


def place(bundle, detached):
    require_plain_directory(detached, "detached input")
    record_path = os.path.join(detached, RECORD_NAME)
    if not is_plain_file(record_path):
        # `signapple apply` is handed the <Bundle>.app inside the detached
        # tree, so accept the same argument and step up to the record.
        detached = os.path.dirname(detached.rstrip("/"))
        require_plain_directory(detached, "detached archive root")
        record_path = os.path.join(detached, RECORD_NAME)
    if not is_plain_file(record_path):
        raise SystemExit(f"placement record is not a regular file: {record_path}")
    with open(record_path) as f:
        record = json.load(f)
    if record.get("schema_version") != 1:
        raise SystemExit(f"unsupported placement record schema: {record.get('schema_version')}")
    validate_detached_root(detached, record)
    root = os.path.join(detached, record["bundle"])
    validate_detached_material(root, record)

    actual_inputs = {}
    for directory, dirs, files in os.walk(bundle):
        for name in dirs + files:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                raise SystemExit(f"unsigned input is a symbolic link: {path}")
        for name in files:
            path = os.path.join(directory, name)
            rel = os.path.relpath(path, bundle).replace(os.sep, "/")
            with open(path, "rb") as f:
                actual_inputs[rel] = f"sha256:{hashlib.sha256(f.read()).hexdigest()}"
    expected_inputs = record["inputs"]
    for rel in sorted(expected_inputs):
        if rel not in actual_inputs:
            raise SystemExit(f"{rel}: unsigned bundle input is missing")
        if actual_inputs[rel] != expected_inputs[rel]:
            raise SystemExit(f"{rel}: unsigned bundle input does not match its record")
    unexpected = sorted(set(actual_inputs) - set(expected_inputs))
    if unexpected:
        raise SystemExit(f"{unexpected[0]}: unexpected unsigned bundle input")

    for rel in sorted(record["machos"]):
        entry = record["machos"][rel]
        path = os.path.join(bundle, rel)
        with open(path, "rb") as f:
            source = f.read()
        expected = entry.get("input_sha256")
        if expected is not None and hashlib.sha256(source).hexdigest() != expected:
            raise SystemExit(f"{rel}: not the build this signature was detached from")

        name = os.path.basename(rel)
        signatures = {}
        for target in entry["slices"]:
            sig = os.path.join(root, "Contents", "MacOS", f"{name}.{ARCH_SUFFIX[target['arch']]}sign")
            with open(sig, "rb") as f:
                signatures[target["arch"]] = f.read()

        data = rebuild(source, facts(path)["slices"], entry, signatures)
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["output_sha256"]:
            raise SystemExit(
                f"{rel}: placed {len(data)} bytes hashing {digest}, "
                f"record says {entry['output_len']} bytes hashing {entry['output_sha256']}"
            )
        os.chmod(path, os.stat(path).st_mode | 0o200)
        with open(path, "wb") as f:
            f.write(data)

    # The bundle seal and, when one has been stapled, the ticket. Both are
    # plain files to `apply`. A seal the record does not name must not survive
    # from whatever signed the input, but other exact-bound files beside it do.
    seal = os.path.join(bundle, "Contents", "_CodeSignature", "CodeResources")
    if "Contents/_CodeSignature/CodeResources" not in record["files"] and os.path.isfile(seal):
        os.remove(seal)
    ticket = os.path.join(bundle, "Contents", "CodeResources")
    if "Contents/CodeResources" not in record["files"] and os.path.isfile(ticket):
        os.remove(ticket)
    for rel, expected in sorted(record["files"].items()):
        src = os.path.join(root, rel)
        dest = os.path.join(bundle, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
        with open(dest, "rb") as f:
            digest = "sha256:" + hashlib.sha256(f.read()).hexdigest()
        if digest != expected:
            raise SystemExit(f"{rel}: {digest}, record says {expected}")

    return bundle


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(place(sys.argv[1], sys.argv[2]))
