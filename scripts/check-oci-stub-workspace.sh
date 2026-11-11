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
#   - it is found by its cargo invocations, and must copy the workspace
#     manifests and carry a recognizable stub step (either missing is an
#     error, never a skip); the stub step is extracted verbatim and run over a
#     copy of the manifests;
#   - `cargo fetch --locked` runs inside that Containerfile's own base image
#     (its first `FROM`), so manifest parsing uses the cargo the image build
#     uses, not this checkout's `rust-toolchain.toml`.
#
# A Containerfile that builds through the repository flake instead (`nix
# build`, the inference node's image) has the same blind spot in another
# form: the flake reads files the Containerfile has to copy in, and a file it
# reads but the image does not copy breaks only the image build. Such a
# Containerfile must carry an `eval` stage that evaluates what it builds over
# exactly the context it copies (a missing stage is an error, never a skip),
# and that stage is built here, `.dockerignore` and all.
#
# Needs Docker (with buildx) and network access.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

command -v docker >/dev/null || { echo "error: docker is required" >&2; exit 1; }

status=0
found=0
for containerfile in oci/*/Containerfile; do
  # A workspace image is any Containerfile that invokes cargo on any
  # non-comment line (a `RUN` may continue across lines); identified
  # independently of the manifest-copy line this check protects, so dropping
  # that line is an error rather than a silent skip.
  grep -Ev '^[[:space:]]*#' "$containerfile" | grep -Eq '(^|[^[:alnum:]_-])cargo (fetch|build)' || continue
  found=$((found + 1))
  if ! grep -q '^COPY --parents crates/\*/Cargo.toml' "$containerfile"; then
    echo "error: $containerfile runs cargo but does not copy the workspace manifests the way this check expects" >&2
    status=1
    continue
  fi

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
  echo "error: no Containerfile runs cargo; this check matched nothing" >&2
  exit 1
fi

flake_found=0
for containerfile in oci/*/Containerfile; do
  grep -Ev '^[[:space:]]*#' "$containerfile" | grep -Eq '(^|[^[:alnum:]_-])nix build' || continue
  flake_found=$((flake_found + 1))
  if ! grep -Eq '^FROM [^[:space:]]+ AS eval$' "$containerfile"; then
    echo "error: $containerfile builds through the flake but has no \`eval\` stage this check can build" >&2
    status=1
    continue
  fi
  err="$(mktemp)"
  if docker buildx build --platform linux/amd64 --target eval --progress plain \
      --file "$containerfile" . >/dev/null 2>"$err"; then
    echo "ok: $containerfile evaluates over the context it copies"
  else
    echo "error: $containerfile: the flake does not evaluate over the context it copies:" >&2
    tail -n 40 "$err" | sed 's/^/  /' >&2
    status=1
  fi
  rm -f "$err"
done

if [ "$flake_found" -eq 0 ]; then
  echo "error: no Containerfile builds through the flake; this check's second half matched nothing" >&2
  exit 1
fi
exit "$status"
