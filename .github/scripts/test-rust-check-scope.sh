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
markdown=false
oci=false' "$apple_path"
done

# The teeth of the glob: a script that does not exist yet must already be in
# scope, because an enumeration is what silently drops one.
for future_path in \
  scripts/apple-notarize.py \
  scripts/apple-staple.sh \
  scripts/fixtures/apple-roundtrip/future-case/facts.json; do
  assert_scope 'rust=false
apple=true
markdown=false
oci=false' "$future_path"
done

assert_scope 'rust=false
apple=true
markdown=true
oci=false' scripts/fixtures/apple-roundtrip/README.md
assert_scope 'rust=true
apple=false
markdown=false
oci=true' crates/eidola-apple/src/lib.rs
# The gateway's build checks the engine pins against the committed engine
# deployments, so a change to either must run the cargo gates (and, with
# them, the OCI rehearsal).
for engine_trust_path in \
  releases/trust/engine-enclaves.json \
  deploy/engine/some-model/some-variant/tinfoil-config.yml \
  deploy/engine/some-model/some-variant/deployment.json; do
  assert_scope 'rust=true
apple=false
markdown=false
oci=true' "$engine_trust_path"
done
assert_scope 'rust=false
apple=false
markdown=false
oci=false' scripts/local-client.sh
assert_scope 'rust=false
apple=false
markdown=false
oci=false' scripts/package-gui-app.sh
assert_scope 'rust=true
apple=true
markdown=true
oci=true' .github/scripts/rust-check-scope.sh

# The OCI stub-resolution rehearsal runs for its own script and for the
# Containerfiles it rehearses, without pulling in the cargo gates.
for oci_path in scripts/check-oci-stub-workspace.sh oci/eidola-server-gateway/Containerfile oci/eidola-cli/Containerfile; do
  assert_scope 'rust=false
apple=false
markdown=false
oci=true' "$oci_path"
done
assert_scope 'rust=false
apple=false
markdown=true
oci=false' docs/verification.md
