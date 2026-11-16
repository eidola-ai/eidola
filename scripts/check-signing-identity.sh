#!/usr/bin/env bash
# Refuse to sign a release that the clients updating to it will reject.
#
# A release's manifest signature is checked by the clients of the release
# before it, against the CI identity *they* embed: the pattern in
# `releases/trust/trust-constants.json` as committed at the previous release
# tag, not as committed now. Moving that pin is a rotation that spans two
# releases (releases/README.md → Rotating the CI signing workflow): the
# release that commits a new pattern is still signed under the old one, and
# only the release after it is signed under the new one. This check is what
# holds that order. The signing job's own identity has to match the
# previous release's pin and be bound to the tag being released, or nothing
# is signed and the release stops here, before any client can see it.
#
#   scripts/check-signing-identity.sh IDENTITY TAG [REPO]
#
# IDENTITY is the Fulcio SAN the signing job's token carries,
# `$GITHUB_SERVER_URL/$GITHUB_WORKFLOW_REF` for a top-level workflow. TAG is
# the tag being released. REPO (default: this repository) must have every
# release tag fetched. "The previous release" is the highest `v*` tag below
# TAG by version, which is the release whose clients update to TAG; with no
# such tag there is no installed client to satisfy, and the check passes.
# The pattern is matched as the client matches it (`*` is any run of
# characters, everything else literal, anchored at both ends).
#
# Bash, git and python3 (stdlib only); no network.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 IDENTITY TAG [REPO]" >&2
  exit 2
fi
identity="$1"
tag="$2"
repo="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
constants="releases/trust/trust-constants.json"

case "$identity" in
  *"@refs/tags/$tag") ;;
  *)
    echo "error: the signing identity \`$identity\` is not bound to the tag being released ($tag)" >&2
    exit 1
    ;;
esac

previous="$(git -C "$repo" tag --list 'v*' | python3 -c '
import re, sys

def key(tag):
    m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?", tag)
    if not m:
        return None
    major, minor, patch, pre = m.groups()
    # A pre-release sorts below its release (v1.2.3-rc1 < v1.2.3).
    return (int(major), int(minor), int(patch), pre is None, pre or "")

current = key(sys.argv[1])
if current is None:
    sys.exit("error: %s is not a release tag (vMAJOR.MINOR.PATCH[-PRE])" % sys.argv[1])
earlier = [t for t in sys.stdin.read().split() if key(t) is not None and key(t) < current]
print(max(earlier, key=key) if earlier else "")
' "$tag")"

if [ -z "$previous" ]; then
  echo "No release precedes $tag: no installed client pins a signing identity yet."
  exit 0
fi

if ! pins="$(git -C "$repo" show "$previous:$constants" 2>/dev/null)"; then
  echo "error: $previous has no $constants, so the identity its clients pin cannot be read" >&2
  exit 1
fi

pattern="$(printf '%s' "$pins" | python3 -c '
import json, sys
value = json.load(sys.stdin).get("expected_ci_identity_pattern")
if not isinstance(value, str) or not value:
    sys.exit("error: no expected_ci_identity_pattern in the previous release'"'"'s trust constants")
print(value)
')"

if python3 -c '
import re, sys
pattern, identity = sys.argv[1], sys.argv[2]
regex = ".*".join(re.escape(part) for part in pattern.split("*"))
sys.exit(0 if re.fullmatch(regex, identity, re.DOTALL) else 1)
' "$pattern" "$identity"; then
  echo "$previous pins \`$pattern\`; \`$identity\` matches it."
else
  cat >&2 <<EOF
error: clients of $previous would reject this release's signature.
  they pin:     $pattern
  this job is:  $identity
A release is signed under the identity the release before it pins
(releases/README.md → Rotating the CI signing workflow). If the pin moved
in this release, sign it from the workflow the old pin names, and move the
signing to the new workflow only after this release is out.
EOF
  exit 1
fi
