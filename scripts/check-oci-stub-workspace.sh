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
# `main` — is the first place that would notice. So, for each such
# Containerfile:
#   - the stub step is extracted verbatim from it (a Containerfile that copies
#     the workspace manifests but has no recognizable stub step is an error,
#     never a skip), and run over a copy of the manifests;
#   - `cargo fetch --locked` runs inside that Containerfile's own base image
#     (its first `FROM`), so manifest parsing uses the cargo the image build
#     uses, not this checkout's `rust-toolchain.toml`.
#
# Needs Docker and network access.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

command -v docker >/dev/null || { echo "error: docker is required" >&2; exit 1; }

status=0
found=0
for containerfile in oci/*/Containerfile; do
  grep -q '^COPY --parents crates/\*/Cargo.toml' "$containerfile" || continue
  found=$((found + 1))

  # The stub step: from `RUN find . -name Cargo.toml` through its last
  # backslash-continued line.
  stub="$(awk '/^RUN find \. -name Cargo\.toml/{p=1} p{print} p&&!/\\$/{exit}' "$containerfile")"
  if [ -z "$stub" ]; then
    echo "error: $containerfile copies the workspace manifests but has no stub step this check recognizes" >&2
    status=1
    continue
  fi
  image="$(awk '/^FROM /{print $2; exit}' "$containerfile")"

  work="$(mktemp -d)"
  cp Cargo.toml Cargo.lock rust-toolchain.toml "$work/"
  for manifest in crates/*/Cargo.toml; do
    mkdir -p "$work/$(dirname "$manifest")"
    cp "$manifest" "$work/$manifest"
  done
  (cd "$work" && sh -c "${stub#RUN }")

  if docker run --rm --platform linux/amd64 --entrypoint cargo \
      -v "$work:/src" -w /src "$image" fetch --locked >/dev/null 2>"$work.err"; then
    echo "ok: $containerfile resolves over stub sources ($image)"
  else
    echo "error: $containerfile: cargo in $image cannot resolve the stubbed workspace:" >&2
    sed 's/^/  /' "$work.err" >&2
    echo "  (a target cargo infers from disk needs an explicit \`path\` in its Cargo.toml)" >&2
    status=1
  fi
  rm -rf "$work" "$work.err"
done

if [ "$found" -eq 0 ]; then
  echo "error: no Containerfile copies the workspace manifests; this check matched nothing" >&2
  exit 1
fi
exit "$status"
