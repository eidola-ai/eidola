set dotenv-load := true

# List available recipes
default:
    @just --list

# --- Development ---

# Start postgres + server (full stack in containers)
dev:
    ./scripts/dev.sh

# Start backing services (postgres) for running the server on the host with cargo
services:
    docker buildx bake postgres
    docker compose up -d --no-build postgres

# Drop and recreate the eidolons database, then apply schema.sql
db-reset:
    docker compose exec postgres dropdb -U eidolons --if-exists eidolons
    docker compose exec postgres createdb -U eidolons eidolons
    docker compose exec postgres psql -U eidolons -d eidolons -f /docker-entrypoint-initdb.d/schema.sql

# --- Rust (inner loop, runs on host) ---

# Lint and format check
check:
    cargo clippy --all-targets -- -D warnings
    cargo fmt --check

# Run all tests
test:
    cargo test

# Run integration tests (requires: just services && just db-reset)
test-integration:
    DATABASE_URL="${DATABASE_URL:-postgres://eidolons@localhost/eidolons}" CREDENTIAL_MASTER_KEY="${CREDENTIAL_MASTER_KEY:-0000000000000000000000000000000000000000000000000000000000000000}" cargo test -p eidolons-server -- --ignored

# Run E2E webhook smoke tests (requires STRIPE_API_KEY)
test-webhook-smoke:
    ./scripts/test-webhook-smoke.sh

# --- Codegen ---

# Regenerate UniFFI Swift bindings and Crux types
update-bindings:
    ./scripts/update-shared-bindings.sh

# Regenerate OpenAPI spec
update-openapi:
    ./scripts/update-server-openapi.sh

# Rebuild XCFramework (dev, native arch only)
update-xcframework:
    ./scripts/update-shared-xcframework-dev.sh

# Rebuild XCFramework (release, universal binary)
update-xcframework-release:
    ./scripts/update-shared-xcframework.sh

# --- CI / Release ---

build:
    # Build OCI images (server, postgresql)
    docker buildx bake

    # TODO: Build apps

# Update artifact-manifest.json with current build digests
# Uses a docker-container builder for rewrite-timestamp + force-compression
# (the default docker driver does not support these reproducibility options).
update-manifest:
    #!/usr/bin/env bash
    set -euo pipefail
    BUILDER="eidolons"
    if ! docker buildx inspect "$BUILDER" &>/dev/null; then
      echo "Creating docker-container builder '$BUILDER'..."
      docker buildx create --name "$BUILDER" --driver docker-container --driver-opt image=moby/buildkit:v0.28.0
    fi
    docker buildx bake \
      --builder "$BUILDER" \
      --set '*.output=type=docker,rewrite-timestamp=true,force-compression=true,compression=gzip' \
      --metadata-file /tmp/bake-metadata.json
    jq -n \
      --arg server "$(jq -r '."server"."containerimage.digest"' /tmp/bake-metadata.json)" \
      --arg postgres "$(jq -r '."postgres"."containerimage.digest"' /tmp/bake-metadata.json)" \
      '{"eidolons-server": $server, "eidolons-postgres": $postgres}' \
      > artifact-manifest.json
    echo "Updated artifact-manifest.json"

# Run all Nix checks (formatting, linting, tests, artifact freshness)
ci-check:
    nix flake check --show-trace

# Build XCFramework via Nix
ci-build-xcframework:
    nix run '.#update-eidolons-shared-swift-xcframework'
