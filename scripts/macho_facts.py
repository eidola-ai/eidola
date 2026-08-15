#!/usr/bin/env python3
"""Dump the structural facts a code signature moves, per Mach-O slice.

Written for the Apple signing work. The round-trip harness and the
differential test against `eidola-apple` both need to name *which field at
which offset* differs when a signature is added, replaced, or detached —
`cmp` alone only says "byte 4183 differs".

Read-only on purpose: this parser is the instrument the reconstruction is
graded with, so it must never become the implementation. Nothing here
writes bytes.

Emits JSON on stdout: one record per slice with the fat-header placement,
`__LINKEDIT`'s sizing (plus the absolute file offset of each field, so a
diff can be pointed at), and the `LC_CODE_SIGNATURE` blob's location and
hash.

Usage: macho-facts.py <mach-o path>
"""

import hashlib
import json
import struct
import sys

FAT_MAGIC = 0xCAFEBABE
FAT_MAGIC_64 = 0xCAFEBABF
MH_MAGIC_64 = 0xFEEDFACF
LC_SEGMENT_64 = 0x19
LC_CODE_SIGNATURE = 0x1D

CPU_NAMES = {
    (0x0100000C, 0): "arm64",
    (0x0100000C, 2): "arm64e",
    (0x01000007, 3): "x86_64",
}

# A `fat_arch` is 20 bytes with 32-bit offset/size; a `fat_arch_64` is 32
# bytes with 64-bit ones and a trailing reserved word. Both are big-endian.
# The two are parsed rather than FAT_MAGIC_64 rejected because `fat_offset`
# is load-bearing — every reported per-slice fact is read through it — so
# reading a 64-bit table with the 32-bit layout yields a plausible-looking
# zero offset rather than an error.
FAT_ARCH = struct.Struct(">iiIII")
FAT_ARCH_64 = struct.Struct(">iiQQII")


def cpu_name(cputype, cpusubtype):
    return CPU_NAMES.get((cputype, cpusubtype & 0x00FFFFFF), f"{cputype}/{cpusubtype}")


def slice_facts(data, base):
    """Facts for the Mach-O whose header starts at `base`."""
    (magic,) = struct.unpack_from("<I", data, base)
    if magic != MH_MAGIC_64:
        raise SystemExit(f"unsupported Mach-O magic {magic:#x} at {base:#x}")
    cputype, cpusubtype, _filetype, ncmds, _sizeofcmds, flags = struct.unpack_from(
        "<iiIIII", data, base + 4
    )

    facts = {
        "arch": cpu_name(cputype, cpusubtype),
        "header_offset": base,
        "mh_flags": f"{flags:#x}",
        # MH_DYLDLINK|... bit 0x02000000 is MH_PIE; 0x00000004 would be
        # MH_DYLDLINK. The one that matters here is the linker-signed hint
        # carried in the signature's CodeDirectory, not the header — see
        # `codesign -dv` output for `adhoc,linker-signed`.
        "linkedit": None,
        "code_signature": None,
    }

    pos = base + 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, pos)
        if cmd == LC_SEGMENT_64:
            segname = data[pos + 8 : pos + 24].rstrip(b"\x00").decode()
            if segname == "__LINKEDIT":
                vmaddr, vmsize, fileoff, filesize = struct.unpack_from(
                    "<QQQQ", data, pos + 24
                )
                facts["linkedit"] = {
                    "vmaddr": vmaddr,
                    "vmsize": vmsize,
                    "fileoff": fileoff,
                    "filesize": filesize,
                    # Absolute file offsets of each field, so a byte diff can
                    # be attributed by address without re-deriving the walk.
                    "vmsize_field_offset": pos + 32,
                    "fileoff_field_offset": pos + 40,
                    "filesize_field_offset": pos + 48,
                }
        elif cmd == LC_CODE_SIGNATURE:
            dataoff, datasize = struct.unpack_from("<II", data, pos + 8)
            blob = data[base + dataoff : base + dataoff + datasize]
            facts["code_signature"] = {
                "dataoff": dataoff,
                "datasize": datasize,
                "lc_offset": pos,
                "superblob_sha256": hashlib.sha256(blob).hexdigest(),
            }
        pos += cmdsize

    return facts


def facts(path):
    with open(path, "rb") as f:
        data = f.read()

    (be_magic,) = struct.unpack_from(">I", data, 0)
    if be_magic in (FAT_MAGIC, FAT_MAGIC_64):
        (nfat,) = struct.unpack_from(">I", data, 4)
        fat_arch = FAT_ARCH_64 if be_magic == FAT_MAGIC_64 else FAT_ARCH
        slices = []
        for i in range(nfat):
            cputype, cpusubtype, offset, size, align = fat_arch.unpack_from(
                data, 8 + i * fat_arch.size
            )[:5]
            s = slice_facts(data, offset)
            s["fat_offset"] = offset
            s["fat_size"] = size
            s["fat_align"] = align
            slices.append(s)
        kind = "fat"
    else:
        slices = [slice_facts(data, 0)]
        kind = "thin"

    return {
        "path": path,
        "kind": kind,
        "file_size": len(data),
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "slices": slices,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(facts(sys.argv[1]), indent=2))
