#!/usr/bin/env bash
# Holds scripts/weights-hash.py (the standalone recompute of a model's weights
# hash, docs/model-identity.md) to the hash the gateway's own code computes:
# the gateway's catalog fixture lists a real weights directory beside it, and
# its tests (`the_fixtures_weights_directory_hashes_to_its_row`) hold that
# directory to the fixture's sha256. Here the script must reach the same hash,
# verify the directory against the fixture, and refuse each way it can be
# wrong. Python 3.11+ standard library only, no toolchain.
set -euo pipefail

cd "$(dirname "$0")/.."
script=scripts/weights-hash.py
fixture=crates/eidola-server-gateway/tests/fixtures/models/fixture-family/flash.toml
weights=crates/eidola-server-gateway/tests/fixtures/models/fixture-family/weights

expected=$(python3 -I -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["sha256"])' "$fixture")
[ ${#expected} -eq 64 ] || { echo "no sha256 in $fixture" >&2; exit 1; }

echo "── the script's hash is the fixture's ──"
actual=$(python3 -I "$script" hash "$weights")
[ "$actual" = "$expected" ] || { echo "hash $actual, fixture says $expected" >&2; exit 1; }
echo "  $actual"

echo "── the sha256sum procedure agrees ──"
if command -v sha256sum >/dev/null; then sum=(sha256sum); else sum=(shasum -a 256); fi
piped=$(cd "$weights" && LC_ALL=C ls *.safetensors config.json model.safetensors.index.json \
    tokenizer.json chat_template.jinja generation_config.json 2>/dev/null \
  | LC_ALL=C sort | xargs "${sum[@]}" | "${sum[@]}" | cut -d' ' -f1)
[ "$piped" = "$expected" ] || { echo "sha256sum procedure gives $piped" >&2; exit 1; }
echo "  ok"

echo "── verify accepts the fixture's directory ──"
python3 -I "$script" verify "$fixture" "$weights" >/dev/null
echo "  ok"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
refused() {
  if python3 -I "$script" verify "$fixture" "$work/w" >"$work/out" 2>&1; then
    echo "verify accepted $1" >&2; exit 1
  fi
  grep -q "$2" "$work/out" || { echo "verify refused $1, but not with \"$2\":" >&2; cat "$work/out" >&2; exit 1; }
  echo "  refused: $1"
}
fresh() { rm -rf "$work/w"; cp -R "$weights" "$work/w"; }

echo "── verify refuses each way a directory can be wrong ──"
fresh; printf 'x' > "$work/w/model-00003-of-00002.safetensors"
refused "an extra shard the engine would hash" "the engine would hash it"
fresh; printf 'shard 1ne' > "$work/w/model-00001-of-00002.safetensors"
refused "a file with other bytes of the same size" "sha256"
fresh; printf 'shard one!' > "$work/w/model-00001-of-00002.safetensors"
refused "a file of another size" "bytes, the catalog says"
fresh; rm "$work/w/tokenizer.json"
refused "a missing file" "tokenizer.json: missing"
fresh; python3 -I -c 'import sys; t = open(sys.argv[1]).read(); open(sys.argv[2], "w").write(t.replace(sys.argv[3], "0" * 64))' "$fixture" "$work/bad.toml" "$expected"
if python3 -I "$script" verify "$work/bad.toml" "$weights" >"$work/out" 2>&1; then
  echo "verify accepted a catalog whose sha256 is not its manifest's" >&2; exit 1
fi
grep -q "the manifest of the listed files hashes to" "$work/out"
echo "  refused: a catalog whose sha256 is not its manifest's aggregate"

echo "weights-hash script: agrees with the gateway's hash, and refuses what it should."
