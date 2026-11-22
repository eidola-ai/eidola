#!/usr/bin/env python3
"""Remove an APK's v2/v3 signing block, restoring the unsigned bytes.

Usage: strip_apk_signature.py SIGNED.apk OUT.apk

A v2+ signature is one block between the last entry and the central
directory. apksigner also zero-pads the entries so the block starts on a
4 KiB boundary; that padding goes too, so the output is the input to signing.
Signing changes nothing else but the EOCD's central-directory offset.
A JAR (v1) signature adds META-INF entries and cannot be stripped this way,
so its presence is an error.
"""
import struct
import sys

MAGIC = b"APK Sig Block 42"


def strip(data: bytes) -> bytes:
    eocd = data.rfind(b"PK\x05\x06")
    if eocd < 0 or len(data) - eocd < 22:
        raise SystemExit("no end of central directory")
    cd_size, cd_off = struct.unpack_from("<II", data, eocd + 12)
    if data[cd_off - 16 : cd_off] != MAGIC:
        raise SystemExit("no APK signing block")
    (size_footer,) = struct.unpack_from("<Q", data, cd_off - 24)
    start = cd_off - size_footer - 8
    (size_header,) = struct.unpack_from("<Q", data, start)
    if size_header != size_footer:
        raise SystemExit("inconsistent signing block sizes")
    cd = data[cd_off : cd_off + cd_size]
    if b"META-INF/" in cd:
        raise SystemExit("JAR signature entries present; not a v2+-only APK")
    end = entries_end(data, cd)
    if end > start or any(data[end:start]):
        raise SystemExit("unexpected bytes between the last entry and the signing block")
    tail = bytearray(data[eocd:])
    struct.pack_into("<I", tail, 16, end)
    return data[:end] + cd + bytes(tail)


def entries_end(data: bytes, cd: bytes) -> int:
    """End of the last local entry (header, name, extra, data), from the central directory."""
    end, i = 0, 0
    while i < len(cd):
        if cd[i : i + 4] != b"PK\x01\x02":
            raise SystemExit("bad central directory")
        flags, = struct.unpack_from("<H", cd, i + 8)
        csize, = struct.unpack_from("<I", cd, i + 20)
        n, m, k = struct.unpack_from("<HHH", cd, i + 28)
        off, = struct.unpack_from("<I", cd, i + 42)
        if flags & 0x8:
            raise SystemExit("data descriptors are not supported")
        ln, lm = struct.unpack_from("<HH", data, off + 26)
        end = max(end, off + 30 + ln + lm + csize)
        i += 46 + n + m + k
    return end


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        out = strip(f.read())
    with open(dst, "wb") as f:
        f.write(out)


if __name__ == "__main__":
    main()
