variable "SOURCE_DATE_EPOCH" {
  default = "1"
}

variable "REGISTRY" {
  default = ""
}

# Comma-separated list of tags (env vars are strings, so we split in HCL)
variable "TAGS" {
  default = "dev"
}

# ── Base target: shared settings applied to all images ─────────────────────────

target "_common" {
  platforms = ["linux/amd64"]
  args = {
    SOURCE_DATE_EPOCH = "${SOURCE_DATE_EPOCH}"
  }
  # Disable provenance attestations (non-deterministic metadata)
  attest = []
}

# ── Local dev targets (compose.yaml overlay) ──────────────────────────────────
# context and dockerfile come from compose.yaml; not repeated here.
# The docker driver does not support rewrite-timestamp or force-compression,
# so local builds omit them. Once all base images are published to a registry,
# switch to a docker-container builder to enable full local reproducibility.

target "server" {
  inherits = ["_common"]
  tags     = ["eidolons-server:dev"]
  cache-from = ["type=local,src=.buildx-cache/server"]
  cache-to   = ["type=local,dest=.buildx-cache/server,mode=max"]
}

target "postgres" {
  inherits = ["_common"]
  tags     = ["eidolons-postgres:dev"]
}

group "default" {
  targets = ["server", "postgres"]
}

# ── CI targets (registry push) ────────────────────────────────────────────────
# These repeat context/dockerfile because they need a different output type
# (push to registry vs load to local daemon).

# Full reproducibility options require the docker-container driver
# (used by CI via setup-buildx-action).
target "_ci" {
  inherits = ["_common"]
  output   = ["type=image,push=true,rewrite-timestamp=true,force-compression=true"]
}

target "ci-server" {
  inherits   = ["_ci"]
  context    = "."
  dockerfile = "crates/eidolons-server/Containerfile"
  tags       = [for t in split(",", TAGS) : "${REGISTRY}/eidolons-server:${t}"]
}

target "ci-postgres" {
  inherits   = ["_ci"]
  context    = "."
  dockerfile = "docker/postgresql/Containerfile"
  tags       = [for t in split(",", TAGS) : "${REGISTRY}/eidolons-postgres:${t}"]
}

group "ci" {
  targets = ["ci-server", "ci-postgres"]
}
