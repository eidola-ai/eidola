variable "SOURCE_DATE_EPOCH" {
  default = "0"
}

variable "REGISTRY" {
  default = ""
}

# Comma-separated list of tags (env vars are strings, so we split in HCL)
variable "TAGS" {
  default = "dev"
}

# Registry namespace (e.g. ghcr.io/<owner>) for the shared BuildKit layer cache.
# Empty (the default, and what local builds use) disables the cache entirely;
# CI sets it for non-release builds and leaves it empty for v* release tags so
# the attested artifacts compile clean. The dev and ci-* variant of each image
# share one ref per image (eidola-buildcache:{server,cli,postgres,server-engine}), so the
# promotion PR (which builds the dev targets) warms the post-merge push to main
# (which builds the ci-* targets). See .github/workflows/artifacts.yml.
variable "CACHE_REGISTRY" {
  default = ""
}

# Cargo profile: "release" for reproducible builds, "docker-dev" for fast iteration
variable "CARGO_PROFILE" {
  default = "release"
}

# The inference node's image root, already built: a directory holding
# `rootfs/` (CI's runner-side Nix build, in Nix's sandbox). Set, it replaces
# the Containerfile's `build` stage as a named context; empty (the default,
# and every local build), that stage runs Nix inside its pinned container.
variable "ENGINE_BUILD_CONTEXT" {
  default = ""
}

# Nix's sandbox in the container build of that root: "true" (the default) or
# "false", which CI's reproduction uses to show the sandbox does not reach
# the image's bytes.
variable "ENGINE_NIX_SANDBOX" {
  default = "true"
}

# ── Base target: shared settings applied to all images ─────────────────────────

target "_common" {
  platforms = ["linux/amd64"]
  args = {
    SOURCE_DATE_EPOCH = "${SOURCE_DATE_EPOCH}"
    CARGO_PROFILE     = "${CARGO_PROFILE}"
  }
  # Disable default inline provenance (contains non-deterministic metadata
  # like builder ID and timestamps that break cross-environment reproducibility).
  # An empty attest list is not enough — buildx still injects mode=min,inline-only
  # provenance unless explicitly disabled.
  attest = ["type=provenance,disabled=true"]
}

# ── Per-image cache bases ──────────────────────────────────────────────────────
# Inherited by both the dev and ci-* variant of each image so they share one
# registry cache ref. `ignore-error=true` keeps cache export non-fatal when the
# run lacks registry write creds (e.g. a fork PR's read-only token). When
# CACHE_REGISTRY is empty (local builds, release tags) both lists are empty —
# no cache import or export.

target "_cache_server" {
  cache-from = CACHE_REGISTRY != "" ? ["type=registry,ref=${CACHE_REGISTRY}/eidola-buildcache:server"] : []
  cache-to   = CACHE_REGISTRY != "" ? ["type=registry,ref=${CACHE_REGISTRY}/eidola-buildcache:server,mode=max,image-manifest=true,oci-mediatypes=true,ignore-error=true"] : []
}

target "_cache_cli" {
  cache-from = CACHE_REGISTRY != "" ? ["type=registry,ref=${CACHE_REGISTRY}/eidola-buildcache:cli"] : []
  cache-to   = CACHE_REGISTRY != "" ? ["type=registry,ref=${CACHE_REGISTRY}/eidola-buildcache:cli,mode=max,image-manifest=true,oci-mediatypes=true,ignore-error=true"] : []
}

target "_cache_postgres" {
  cache-from = CACHE_REGISTRY != "" ? ["type=registry,ref=${CACHE_REGISTRY}/eidola-buildcache:postgres"] : []
  cache-to   = CACHE_REGISTRY != "" ? ["type=registry,ref=${CACHE_REGISTRY}/eidola-buildcache:postgres,mode=max,image-manifest=true,oci-mediatypes=true,ignore-error=true"] : []
}

# The engine's cache holds the container build's layers, so a build from a
# named context neither reads nor writes it: its one step is a copy, and
# writing would replace the container build's layers with that copy.
target "_cache_server_engine" {
  cache-from = CACHE_REGISTRY != "" && ENGINE_BUILD_CONTEXT == "" ? ["type=registry,ref=${CACHE_REGISTRY}/eidola-buildcache:server-engine"] : []
  cache-to   = CACHE_REGISTRY != "" && ENGINE_BUILD_CONTEXT == "" ? ["type=registry,ref=${CACHE_REGISTRY}/eidola-buildcache:server-engine,mode=max,image-manifest=true,oci-mediatypes=true,ignore-error=true"] : []
}

# How the engine image's root is reached (see ENGINE_BUILD_CONTEXT). The
# container build's Nix step is privileged so it can host Nix's sandbox, so
# it requests `security.insecure`, which the builder must allow; a build from
# the named context runs no such step and requests nothing.
target "_engine_root" {
  args = {
    NIX_SANDBOX = ENGINE_NIX_SANDBOX
  }
  contexts     = ENGINE_BUILD_CONTEXT != "" ? { build = ENGINE_BUILD_CONTEXT } : {}
  entitlements = ENGINE_BUILD_CONTEXT != "" ? [] : ["security.insecure"]
}

# ── Local dev targets ─────────────────────────────────────────────────────────
# Every target declares its own context and dockerfile. Bake also reads
# compose.yaml and merges a compose service into the target of the same name,
# but that coupling is by name only: a target leaning on it silently falls
# back to `./Dockerfile` the moment the service is renamed. compose.yaml's
# build sections stay for `docker compose build`; they are not this file's
# source of truth.
# For reproducible builds, use a docker-container builder with OCI output and
# --set '*.output=type=image,push=false,rewrite-timestamp=true,force-compression=true,oci-mediatypes=true'
# (the default docker driver does not support these options).

# The `server` and `ci-server` targets build the gateway image
# (eidola-server-gateway). The target name is not cosmetic:
# scripts/artifact-manifest.sh records each target's digest under the
# manifest key `eidola-<target>`, and installed clients compare that key set,
# so renaming a target is a manifest artifact-set rotation (releases/README.md).
target "server" {
  inherits   = ["_common", "_cache_server"]
  context    = "."
  dockerfile = "oci/eidola-server-gateway/Containerfile"
  tags       = ["eidola-server-gateway:dev"]
}

target "cli" {
  inherits   = ["_common", "_cache_cli"]
  context    = "."
  dockerfile = "oci/eidola-cli/Containerfile"
  tags       = ["eidola-cli:dev"]
}

# Operational image, runnable as receiver, worker or private support CLI.
# Kept outside the client artifact manifest and release publication pipeline.
target "billing" {
  inherits   = ["_common"]
  context    = "."
  dockerfile = "oci/eidola-billing/Containerfile"
  tags       = ["eidola-billing:dev"]
}

target "postgres" {
  inherits   = ["_common", "_cache_postgres"]
  context    = "."
  dockerfile = "oci/postgresql/Containerfile"
  tags       = ["eidola-postgres:dev"]
}

# The inference node (eidola-server-engine, CUDA executor). Recorded under
# the manifest key `eidola-server-engine` from the manifest schema that
# records it (scripts/artifact-manifest.sh, ENGINE_IMAGE_SCHEMA). Its build
# is Nix (on CI's runner, or in the Containerfile's builder stage), so
# CARGO_PROFILE does not reach it: the image is always the release profile.
target "server-engine" {
  inherits   = ["_common", "_cache_server_engine", "_engine_root"]
  context    = "."
  dockerfile = "oci/eidola-server-engine/Containerfile"
  tags       = ["eidola-server-engine:dev"]
}

target "shim" {
  inherits   = ["_common"]
  context    = "."
  dockerfile = "oci/tinfoil-shim-mock/Containerfile"
  tags       = ["tinfoil-shim-mock:dev"]
}

# Stripe CLI — pins the upstream image by digest so dependabot can propose
# updates via the Containerfile, rather than silently pulling :latest.
target "stripe-cli" {
  context    = "."
  dockerfile = "oci/stripe-cli/Containerfile"
  tags       = ["stripe-cli:dev"]
  attest     = []
}

group "default" {
  targets = ["server", "cli", "postgres", "shim"]
}

# ── CI targets (registry push) ────────────────────────────────────────────────
# These repeat context/dockerfile because they need a different output type
# (push to registry vs load to local daemon).

# Full reproducibility options require the docker-container driver
# (used by CI via setup-buildx-action).
target "_ci" {
  inherits = ["_common"]
  output   = ["type=image,push=true,rewrite-timestamp=true,force-compression=true,compression=gzip,oci-mediatypes=true"]
}

target "ci-server" {
  inherits   = ["_ci", "_cache_server"]
  context    = "."
  dockerfile = "oci/eidola-server-gateway/Containerfile"
  tags       = [for t in split(",", TAGS) : "${REGISTRY}/eidola-server-gateway:${t}"]
}

target "ci-cli" {
  inherits   = ["_ci", "_cache_cli"]
  context    = "."
  dockerfile = "oci/eidola-cli/Containerfile"
  tags       = [for t in split(",", TAGS) : "${REGISTRY}/eidola-cli:${t}"]
}

target "ci-postgres" {
  inherits   = ["_ci", "_cache_postgres"]
  context    = "."
  dockerfile = "oci/postgresql/Containerfile"
  tags       = [for t in split(",", TAGS) : "${REGISTRY}/eidola-postgres:${t}"]
}

target "ci-server-engine" {
  inherits   = ["_ci", "_cache_server_engine", "_engine_root"]
  context    = "."
  dockerfile = "oci/eidola-server-engine/Containerfile"
  tags       = [for t in split(",", TAGS) : "${REGISTRY}/eidola-server-engine:${t}"]
}
