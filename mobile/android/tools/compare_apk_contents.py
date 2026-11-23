#!/usr/bin/env python3
"""Compare two APKs file by file, by content: the check for an APK whose
bytes cannot match ours, such as one Play generated from the App Bundle.

    compare_apk_contents.py OURS.apk OTHER.apk [--require PREFIX]...

Prints each entry as `same`, `differs`, `only ours` or `only theirs`, with
signing entries (`META-INF/`, and the `stamp-cert-sha256` Play adds) listed
apart, since a signature is never ours. With `--require`, exits 1 unless every
entry of ours under each PREFIX (or named exactly) is present in OTHER with
the same bytes: `--require classes.dex --require assets/` is the shell's code
and the app bundle it serves.

Standard library only; reads its inputs, writes nothing.
"""

import hashlib
import sys
import zipfile

SIGNING = ("META-INF/", "stamp-cert-sha256")


def contents(path):
    with zipfile.ZipFile(path) as z:
        return {
            i.filename: hashlib.sha256(z.read(i)).hexdigest()
            for i in z.infolist()
            if not i.is_dir()
        }


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    ours, theirs = contents(args[0]), contents(args[1])
    required = []
    rest = args[2:]
    while rest:
        if rest[0] != "--require" or len(rest) < 2:
            raise SystemExit(__doc__)
        required.append(rest[1])
        rest = rest[2:]

    signing = sorted(n for n in set(ours) | set(theirs) if n.startswith(SIGNING))
    for name in sorted(set(ours) | set(theirs), key=str.encode):
        if name in signing:
            continue
        if name not in theirs:
            state = "only ours"
        elif name not in ours:
            state = "only theirs"
        elif ours[name] == theirs[name]:
            state = "same"
        else:
            state = "differs"
        print("{:<12} {}".format(state, name))
    for name in signing:
        print("{:<12} {}".format("signing", name))

    failed = []
    for prefix in required:
        mine = [n for n in ours if n == prefix or (prefix.endswith("/") and n.startswith(prefix))]
        if not mine:
            failed.append("{}: nothing of ours matches".format(prefix))
        for name in mine:
            if theirs.get(name) != ours[name]:
                failed.append("{}: not the same bytes".format(name))
    for line in failed:
        print("REQUIRED " + line, file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
