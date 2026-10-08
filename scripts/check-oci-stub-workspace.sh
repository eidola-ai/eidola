#!/usr/bin/env bash
# Rehearses the dependency-fetch stage of every OCI Containerfile that builds
# the Rust workspace from manifests plus stub sources, and fails if cargo
# cannot resolve the stubbed workspace.
#
# Those stages copy only `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml` and
# `crates/*/Cargo.toml`, then run a `find … touch src/lib.rs` stub step before
# `cargo fetch --locked`. A target whose path cargo has to infer from files on
# disk (an `[[example]]`, `[[test]]` or `[[bench]]` without `path`) resolves in
# a full checkout but not over stubs, and the image build — which only runs on
# `main` — is the first place that would notice. This runs the stub step
# extracted verbatim from each Containerfile, so the rehearsal cannot drift
# from the recipe it stands in for.
#
# Needs network access for `cargo fetch` (a warm registry cache makes it fast).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

status=0
found=0
for containerfile in oci/*/Containerfile; do
  # The stub step: from `RUN find . -name Cargo.toml` through its last
  # backslash-continued line.
  stub="$(awk '/^RUN find \. -name Cargo\.toml/{p=1} p{print} p&&!/\\$/{exit}' "$containerfile")"
  [ -n "$stub" ] || continue
  found=$((found + 1))

  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  cp Cargo.toml Cargo.lock rust-toolchain.toml "$work/"
  for manifest in crates/*/Cargo.toml; do
    mkdir -p "$work/$(dirname "$manifest")"
    cp "$manifest" "$work/$manifest"
  done
  (cd "$work" && sh -c "${stub#RUN }")

  if (cd "$work" && cargo fetch --locked >/dev/null 2>"$work/fetch.err"); then
    echo "ok: $containerfile resolves over stub sources"
  else
    echo "error: $containerfile: cargo cannot resolve the stubbed workspace:" >&2
    sed 's/^/  /' "$work/fetch.err" >&2
    echo "  (give the target an explicit \`path\` in its Cargo.toml)" >&2
    status=1
  fi
  rm -rf "$work"
  trap - EXIT
done

if [ "$found" -eq 0 ]; then
  echo "error: no Containerfile has the stub step; this check matched nothing" >&2
  exit 1
fi
exit "$status"
