#!/usr/bin/env python3
"""Rewrite a zip with every entry stored, in name order, with one fixed
timestamp and no extra fields.

    zip_stored.py IN.zip OUT.zip

bundletool deflates the App Bundle's entries with the JDK's zlib, so those
bytes would depend on which zlib the JDK running it carries. Stored entries
depend on nothing but their contents, as the APK's do (tools/ApkZip.java).
The entries and their contents are unchanged; Play reads a bundle's entries
whatever their compression, and an asset's compression in the APKs Play
generates is set by the bundle's own configuration, not by this zip's.

Standard library only; reads IN.zip and writes OUT.zip, nothing else.
"""

import sys
import zipfile

# 2000-01-01T00:00:00, the APK's timestamp.
FIXED = (2000, 1, 1, 0, 0, 0)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1:]
    with zipfile.ZipFile(src) as z:
        entries = {}
        for info in z.infolist():
            if info.is_dir():
                continue
            if info.filename in entries:
                raise SystemExit("duplicate entry: " + info.filename)
            entries[info.filename] = z.read(info)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for name in sorted(entries, key=str.encode):
            info = zipfile.ZipInfo(name, FIXED)
            info.external_attr = 0o644 << 16
            info.create_system = 0
            z.writestr(info, entries[name])


if __name__ == "__main__":
    main()
