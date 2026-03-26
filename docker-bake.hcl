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

# Cargo profile: "release" for reproducible builds, "docker-dev" for fast iteration
variable "CARGO_PROFILE" {
  default = "release"
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

# ── Local dev targets (compose.yaml overlay) ──────────────────────────────────
# context and dockerfile come from compose.yaml; not repeated here.
# For reproducible builds, use a docker-container builder and pass
# --set '*.output=type=docker,rewrite-timestamp=true,force-compression=true'
# (the default docker driver does not support these options).

target "server" {
  inherits = ["_common"]
  tags     = ["eidola-server:dev"]
}

target "cli" {
  inherits   = ["_common"]
  context    = "."
  dockerfile = "oci/eidola-cli/Containerfile"
  tags       = ["eidola-cli:dev"]
}

target "postgres" {
  inherits = ["_common"]
  tags     = ["eidola-postgres:dev"]
}

target "shim" {
  inherits = ["_common"]
  tags     = ["tinfoil-shim-mock:dev"]
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

group "manifest" {
  targets = ["server", "cli", "postgres"]
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
  inherits   = ["_ci"]
  context    = "."
  dockerfile = "oci/eidola-server/Containerfile"
  tags       = [for t in split(",", TAGS) : "${REGISTRY}/eidola-server:${t}"]
}

target "ci-cli" {
  inherits   = ["_ci"]
  context    = "."
  dockerfile = "oci/eidola-cli/Containerfile"
  tags       = [for t in split(",", TAGS) : "${REGISTRY}/eidola-cli:${t}"]
}

target "ci-postgres" {
  inherits   = ["_ci"]
  context    = "."
  dockerfile = "oci/postgresql/Containerfile"
  tags       = [for t in split(",", TAGS) : "${REGISTRY}/eidola-postgres:${t}"]
}

group "ci" {
  targets = ["ci-server", "ci-cli", "ci-postgres"]
}
