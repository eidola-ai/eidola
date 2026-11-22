#!/usr/bin/env python3
"""Prints the bundle digest a shell shows for a bundle directory (shell-contract.md →
Bundle digest), and fails on a file a shell would not serve.

Usage: bundle_digest.py DIR
"""
import hashlib
import os
import re
import sys

SEGMENT = re.compile(r"[A-Za-z0-9._-]+")


def digest(root: str) -> str:
    paths = []
    for base, dirs, files in os.walk(root):
        for name in files:
            rel = os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/")
            if not all(SEGMENT.fullmatch(s) and not s.startswith(".") for s in rel.split("/")):
                raise SystemExit(f"{rel}: not a bundle path")
            paths.append(rel)
    manifest = b"".join(
        b"%s  %s\n" % (hashlib.sha256(open(os.path.join(root, p), "rb").read()).hexdigest().encode(), p.encode())
        for p in sorted(paths, key=str.encode)
    )
    return hashlib.sha256(manifest).hexdigest()


if __name__ == "__main__":
    print(digest(sys.argv[1]))
