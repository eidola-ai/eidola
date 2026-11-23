#!/usr/bin/env python3
"""Pack a bundle directory into the container a release publishes it in, and
print the bundle's artifact-manifest.json row.

    bundle_container.py DIR OUT.tar

The container is an uncompressed POSIX ustar archive of the bundle's regular
files, in path order, with fixed metadata (mode 0644, owner 0, time 0) and no
directory entries, ending in two zero blocks: the bytes
`updater::mobile::container::write_ustar` writes for the same files, and the
shape its reader accepts. The row (stdout) is the manifest's `bundle` row:
the container's sha256, the bundle digest (shell-contract.md, Bundle digest)
and every file's path, size and sha256.

Standard library only; reads DIR and writes OUT.tar, nothing else.
"""

import hashlib
import json
import os
import sys

BLOCK = 512


def octal(value, width):
    # Zero-padded, filling the field but its last byte, which is NUL.
    return ("%0*o" % (width - 1, value)).encode() + b"\0"


def header(path, size):
    name, prefix = path.encode(), b""
    if len(name) > 100:
        cut = path.rfind("/", 0, 156)
        if cut <= 0 or len(path) - cut - 1 > 100:
            raise SystemExit("too long for a ustar header: " + path)
        prefix, name = path[:cut].encode(), path[cut + 1 :].encode()
    h = bytearray(BLOCK)
    h[0 : len(name)] = name
    h[100:108] = octal(0o644, 8)
    h[108:116] = octal(0, 8)
    h[116:124] = octal(0, 8)
    h[124:136] = octal(size, 12)
    h[136:148] = octal(0, 12)
    h[156:157] = b"0"
    h[257:263] = b"ustar\0"
    h[263:265] = b"00"
    h[345 : 345 + len(prefix)] = prefix
    h[148:156] = b" " * 8
    h[148:156] = ("%06o" % sum(h)).encode() + b"\0 "
    return bytes(h)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    root, out = sys.argv[1], sys.argv[2]
    paths = []
    for base, dirs, names in os.walk(root):
        dirs.sort()
        for name in names:
            full = os.path.join(base, name)
            if not os.path.isfile(full) or os.path.islink(full):
                raise SystemExit("not a regular file: " + full)
            paths.append(os.path.relpath(full, root).replace(os.sep, "/"))
    paths.sort(key=lambda p: p.encode())

    files, container = [], bytearray()
    for path in paths:
        with open(os.path.join(root, path), "rb") as handle:
            data = handle.read()
        container += header(path, len(data)) + data
        container += b"\0" * (-len(container) % BLOCK)
        files.append(
            {"path": path, "size": len(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}
        )
    container += b"\0" * (2 * BLOCK)
    with open(out, "wb") as handle:
        handle.write(container)

    lines = "".join(f["sha256"][len("sha256:") :] + "  " + f["path"] + "\n" for f in files)
    print(
        json.dumps(
            {
                "type": "bundle",
                "platform": "webview/wasm32",
                "sha256": "sha256:" + hashlib.sha256(container).hexdigest(),
                "digest": "sha256:" + hashlib.sha256(lines.encode()).hexdigest(),
                "files": files,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
