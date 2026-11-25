#!/usr/bin/env python3
"""Write the base module bundletool builds the App Bundle (AAB) from.

    aab_module.py OUT.zip PROTO.apk CLASSES.dex BUNDLE_DIR

PROTO.apk is `aapt2 link --proto-format` over the same manifest and resources
the APK links. The module is laid out as bundletool expects:
`manifest/AndroidManifest.xml`, `resources.pb` and `res/` from PROTO.apk,
`dex/classes.dex`, and the bundle under `assets/bundle/`, every entry stored,
in name order, with one fixed timestamp, so the module is a function of its
inputs. The APK's own zip (tools/ApkZip.java) takes the same files.

Standard library only; reads its inputs and writes OUT.zip, nothing else.
"""

import os
import sys
import zipfile

# 2000-01-01T00:00:00, the APK's timestamp.
FIXED = (2000, 1, 1, 0, 0, 0)


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    out, proto, dex, bundle = sys.argv[1:]
    entries = {}
    with zipfile.ZipFile(proto) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name == "AndroidManifest.xml":
                name = "manifest/AndroidManifest.xml"
            elif name != "resources.pb" and not name.startswith("res/"):
                raise SystemExit("unexpected entry in the proto APK: " + name)
            entries[name] = z.read(info)
    with open(dex, "rb") as handle:
        entries["dex/classes.dex"] = handle.read()
    for base, dirs, names in os.walk(bundle):
        dirs.sort()
        for name in names:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, bundle).replace(os.sep, "/")
            # The APK's zip writer skips dot files the same way.
            if rel.startswith(".") or "/." in rel:
                continue
            with open(full, "rb") as handle:
                entries["assets/bundle/" + rel] = handle.read()
    if "manifest/AndroidManifest.xml" not in entries or "resources.pb" not in entries:
        raise SystemExit("the proto APK has no manifest or resource table")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for name in sorted(entries, key=str.encode):
            info = zipfile.ZipInfo(name, FIXED)
            info.external_attr = 0o644 << 16
            info.create_system = 0
            z.writestr(info, entries[name])


if __name__ == "__main__":
    main()
