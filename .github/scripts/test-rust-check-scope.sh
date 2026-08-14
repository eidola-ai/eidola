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

assert_scope 'rust=false
apple=true
markdown=false' scripts/apple-detach.py
assert_scope 'rust=false
apple=true
markdown=true' scripts/fixtures/apple-roundtrip/README.md
assert_scope 'rust=true
apple=false
markdown=false' crates/eidola-apple/src/lib.rs
assert_scope 'rust=false
apple=false
markdown=false' scripts/local-client.sh
assert_scope 'rust=true
apple=true
markdown=true' .github/scripts/rust-check-scope.sh
