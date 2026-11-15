#!/usr/bin/env bash
# Drives scripts/platform-endorsements-drift.py through each verdict it can
# reach. The script runs only on a schedule against the default branch, so a
# change that broke it would otherwise surface as a silent green or a red run
# nobody can act on. Its notion of a deployment that is behind must be the
# gateway build's (`require_vendored_releases_named`): older than the newest
# release, and drift unless the sidecar states a hold. Python 3 standard
# library, no toolchain.
set -euo pipefail

cd "$(dirname "$0")/.."
script=scripts/platform-endorsements-drift.py
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# A tree vendoring the given releases, with one deployment naming $1 and
# holding for $2 (empty: no hold member; "-": an empty hold).
tree() {
  local release=$1 hold=$2
  shift 2
  rm -rf "$tmp/root"
  mkdir -p "$tmp/root/deploy/engine/m/v"
  for tag in "$@"; do mkdir -p "$tmp/root/releases/trust/platform-endorsements/$tag"; done
  local member=""
  case $hold in
    "") ;;
    -) member=', "hold": "  "' ;;
    *) member=", \"hold\": \"$hold\"" ;;
  esac
  printf '{"tdx": {"platform_endorsements": {"release": "%s"%s}}}\n' "$release" "$member" \
    > "$tmp/root/deploy/engine/m/v/deployment.json"
}

expect() {
  local want=$1 what=$2 upstream=$3
  printf '%s\n' $upstream > "$tmp/tags.txt"
  local got=0
  python3 -I "$script" "$tmp/root" "$tmp/tags.txt" > "$tmp/out" 2>&1 || got=$?
  if [ "$got" != "$want" ]; then
    echo "$what: exit $got, expected $want" >&2
    cat "$tmp/out" >&2
    exit 1
  fi
  echo "  ok: $what (exit $got)"
}

echo "── verdicts ──"
tree platform-v0.1.11 "" platform-v0.1.11
expect 0 "deployment on the newest release, nothing newer" "v0.15.3 platform-v0.1.11 platform-v0.1.9"
expect 1 "a newer release than the newest vendored" "platform-v0.1.11 platform-v0.1.12"

tree platform-v0.1.11 "" platform-v0.1.11 platform-v0.1.12
expect 1 "a deployment behind with no hold" "platform-v0.1.12"

tree platform-v0.1.11 "keeps a retired host until it is replaced" platform-v0.1.11 platform-v0.1.12
expect 0 "a deployment behind with a hold" "platform-v0.1.12"
grep -q '::notice::.*keeps a retired host' "$tmp/out" || { echo "the hold is not reported" >&2; exit 1; }
expect 1 "a held deployment, and a newer release than the newest vendored" \
  "platform-v0.1.12 platform-v0.1.13"

tree platform-v0.1.11 - platform-v0.1.11 platform-v0.1.12
expect 1 "an empty hold" "platform-v0.1.12"

tree platform-v0.1.11 "" platform-v0.1.11
expect 2 "no platform release upstream" "v0.15.3 v0.15.2"
tree platform-v0.1.11 ""
expect 2 "nothing vendored" "platform-v0.1.11"

echo "── the committed tree, against its own newest release ──"
newest=$(python3 -I -c '
import pathlib, sys
tags = [p.name for p in pathlib.Path(sys.argv[1]).iterdir()]
print(max(tags, key=lambda t: tuple(int(n) for n in t.removeprefix("platform-v").split("."))))
' releases/trust/platform-endorsements)
printf '%s\n' "$newest" > "$tmp/tags.txt"
python3 -I "$script" . "$tmp/tags.txt"
