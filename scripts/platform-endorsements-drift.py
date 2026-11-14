"""Report when the platform provider has published platform endorsements
newer than the ones this tree vendors or its engine deployments name.

usage: python3 -I scripts/platform-endorsements-drift.py <root> <upstream-tags>

<root> is the workspace root; <upstream-tags> a file of the provider's
release tags, one per line (any tag; only `platform-vX.Y.Z` ones count).
Exits 1 on drift, 2 when either side lists no platform release at all (a
check that found nothing to compare must not pass), 0 otherwise. Standard
library only.
"""

import json
import pathlib
import re
import sys

TAG = re.compile(r"^platform-v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
VENDORED = "releases/trust/platform-endorsements"


def version(tag):
    match = TAG.match(tag)
    return tuple(int(n) for n in match.groups()) if match else None


def main(root, upstream_file):
    root = pathlib.Path(root)
    with open(upstream_file) as lines:
        upstream = sorted(
            {line.strip() for line in lines if version(line.strip())},
            key=version,
        )
    vendored_dir = root / VENDORED
    vendored = sorted(
        (p.name for p in vendored_dir.iterdir() if p.is_dir() and version(p.name))
        if vendored_dir.is_dir()
        else [],
        key=version,
    )
    if not upstream:
        print("no platform-vX.Y.Z release found upstream; the listing is not what was expected")
        return 2
    if not vendored:
        print(f"no platform-vX.Y.Z release vendored under {VENDORED}")
        return 2
    newest = upstream[-1]
    print(f"newest upstream platform release: {newest} ({len(upstream)} listed)")
    print(f"newest vendored platform release: {vendored[-1]}")

    drift = []
    if version(newest) > version(vendored[-1]):
        drift.append(
            f"{newest} is published but the newest vendored release is {vendored[-1]}: "
            f"vendor it with `just engine-platform-endorsements {newest}` and review "
            "what changed (raised floors, machines added or removed)"
        )
    for sidecar in sorted((root / "deploy/engine").glob("*/*/deployment.json")):
        release = (
            json.loads(sidecar.read_text())
            .get("tdx", {})
            .get("platform_endorsements", {})
            .get("release")
        )
        at = sidecar.relative_to(root)
        if release is None:
            continue
        print(f"{at}: {release}")
        if version(release) is None or version(release) < version(newest):
            drift.append(f"{at} names {release}; {newest} is published")
    for line in drift:
        print(f"::error::{line}")
    return 1 if drift else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
