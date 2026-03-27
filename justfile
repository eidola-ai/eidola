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

# Drop and recreate the eidola database, then apply schema.sql
db-reset:
    docker compose exec postgres dropdb -U eidola --if-exists eidola
    docker compose exec postgres createdb -U eidola eidola
    docker compose exec postgres psql -U eidola -d eidola -f /docker-entrypoint-initdb.d/schema.sql

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
    DATABASE_URL="${DATABASE_URL:-postgres://eidola@localhost/eidola}" CREDENTIAL_MASTER_KEY="${CREDENTIAL_MASTER_KEY:-0000000000000000000000000000000000000000000000000000000000000000}" cargo test -p eidola-server -- --ignored

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
    # Build OCI images (server, cli, postgresql)
    docker buildx bake

# Update artifact-manifest.json with current build digests
# Builds the OCI images plus the macOS app/CLI, then records their digests.
# Requires macOS for the Nix-built app and CLI artifacts.
update-manifest:
    ./scripts/artifact-manifest.sh update --ensure-builder

# Run all Nix checks (formatting, linting, tests, artifact freshness)
ci-check:
    nix flake check --show-trace

# Build XCFramework via Nix
ci-build-xcframework:
    nix run '.#update-eidola-shared-swift-xcframework'

# Build macOS app via Nix (reproducible, open-source Swift 6.2 toolchain)
build-macos-app:
    nix build '.#eidola-macos-app' --show-trace
