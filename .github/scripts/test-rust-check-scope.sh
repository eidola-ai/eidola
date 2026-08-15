#!/bin/sh
set -eu

classifier=.github/scripts/rust-check-scope.sh

assert_scope() {
  expected=$1
  shift
  actual=$(printf '%s\n' "$@" | "$classifier")
  if [ "$actual" != "$expected" ]; then
    printf 'scope mismatch for %s\nexpected:\n%s\nactual:\n%s\n' "$*" "$expected" "$actual" >&2
    exit 1
  fi
}

for apple_path in \
  scripts/verify-apple.sh \
  scripts/test-verify-apple.sh \
  scripts/apple-roundtrip.sh \
  scripts/apple-signature-differential.sh \
  scripts/apple_linkedit_diff.py \
  scripts/macho_facts.py \
  scripts/fixtures/apple-roundtrip/synthetic-universal/detached/eidola-placement.json; do
  assert_scope 'rust=false
apple=true
markdown=false' "$apple_path"
done

# The teeth of the glob: a script that does not exist yet must already be in
# scope, because an enumeration is what silently drops one.
for future_path in \
  scripts/apple-notarize.py \
  scripts/apple-staple.sh \
  scripts/fixtures/apple-roundtrip/future-case/facts.json; do
  assert_scope 'rust=false
apple=true
markdown=false' "$future_path"
done

assert_scope 'rust=false
apple=true
markdown=true' scripts/fixtures/apple-roundtrip/README.md
assert_scope 'rust=true
apple=false
markdown=false' crates/eidola-apple/src/lib.rs
assert_scope 'rust=false
apple=false
markdown=false' scripts/local-client.sh
assert_scope 'rust=false
apple=false
markdown=false' scripts/package-gui-app.sh
assert_scope 'rust=true
apple=true
markdown=true' .github/scripts/rust-check-scope.sh
