#!/usr/bin/env bash
# Vendor one release of the platform provider's signed platform endorsements
# (tinfoilsh/cvmimage `platform-vX.Y.Z`) into
# releases/trust/platform-endorsements/<tag>/.
#
# usage: scripts/engine-platform-endorsements.sh platform-vX.Y.Z
#
# This is the only place the endorsements are fetched. The release asset and
# GitHub's attestations for its digest are downloaded into a fresh, empty
# temporary directory as untrusted data; measure-enclave's
# vendor-platform-endorsements then verifies them offline against the
# committed Sigstore trusted root, exactly as the gateway's build will, and
# writes only what verified. Review the resulting diff like any trust-root
# change. Set GH_TOKEN to raise GitHub's API rate limit; nothing else reads it.
set -euo pipefail

TAG="${1:-}"
REPOSITORY="tinfoilsh/cvmimage"
FILE="platform-endorsements.json"

if [[ ! "$TAG" =~ ^platform-v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
  echo "usage: $0 platform-vX.Y.Z" >&2
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | cut -d ' ' -f 1; }
else
  sha256() { shasum -a 256 "$1" | cut -d ' ' -f 1; }
fi

auth=()
if [[ -n "${GH_TOKEN:-}" ]]; then
  auth=(-H "Authorization: Bearer $GH_TOKEN")
fi

echo "Downloading $FILE from $REPOSITORY release $TAG"
curl -fsSL --proto '=https' --tlsv1.2 \
  "https://github.com/$REPOSITORY/releases/download/$TAG/$FILE" \
  -o "$WORK/$FILE"
DIGEST="$(sha256 "$WORK/$FILE")"

echo "Downloading GitHub's attestations for sha256:$DIGEST"
curl -fsSL --proto '=https' --tlsv1.2 ${auth[@]+"${auth[@]}"} \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/$REPOSITORY/attestations/sha256:$DIGEST?per_page=100" \
  -o "$WORK/attestations.json"

cd "$ROOT"
cargo run -q -p measure-enclave --bin vendor-platform-endorsements -- \
  --workspace "$ROOT" \
  --tag "$TAG" \
  --endorsements "$WORK/$FILE" \
  --attestations "$WORK/attestations.json"

echo
git -C "$ROOT" status --short -- releases/trust/platform-endorsements
