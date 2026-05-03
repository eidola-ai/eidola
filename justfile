set dotenv-load := true

# List available recipes
default:
    @just --list

# --- Development ---

# Run the full server stack with the server in a container (detached).
dev:
    ./scripts/dev.sh --container

# Run backing services for host-mode dev — server runs on host with cargo; writes .env.local with BIND_ADDR + STRIPE_WEBHOOK_SECRET to source.
services:
    ./scripts/dev.sh --host

# Stop everything started by `just dev` or `just services`.
down:
    docker compose --profile server --profile stripe down --remove-orphans

# Drop and recreate the eidola database, then apply schema.sql
db-reset:
    docker compose exec postgres dropdb -U eidola --if-exists eidola
    docker compose exec postgres createdb -U eidola eidola
    docker compose exec postgres psql -U eidola -d eidola -f /docker-entrypoint-initdb.d/schema.sql

# --- Build (local toolchain, fast iteration) ---

# Build a system: server, cli, or macos
build system:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{ system }}" in
      server)
        cargo build -p eidola-server
        ;;
      cli)
        cargo build -p eidola-cli
        ;;
      gui)
        cargo build -p eidola-gui
        ;;
      macos)
        if [[ "$(uname -s)" != "Darwin" ]]; then
          echo "error: macOS app can only be built on macOS" >&2
          exit 1
        fi
        just update-bindings
        just update-xcframework
        ( cd apps/macos && swift build )
        ./scripts/package-macos-app.sh
        ;;
      *)
        echo "error: unknown system '{{ system }}' (expected: server, cli, gui, macos)" >&2
        exit 1
        ;;
    esac

# Build and run a system: server, cli, or macos
run system *args:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{ system }}" in
      server)
        cargo run -p eidola-server -- {{ args }}
        ;;
      cli)
        cargo run -p eidola-cli -- {{ args }}
        ;;
      gui)
        cargo run -p eidola-gui -- {{ args }}
        ;;
      macos)
        if [[ "$(uname -s)" != "Darwin" ]]; then
          echo "error: macOS app can only run on macOS" >&2
          exit 1
        fi
        just build macos
        open "apps/macos/build/Eidola.app" {{ args }}
        ;;
      *)
        echo "error: unknown system '{{ system }}' (expected: server, cli, gui, macos)" >&2
        exit 1
        ;;
    esac

# --- Checks & Tests (inner loop, runs on host) ---

# Lint and format check
check:
    cargo clippy --all-targets -- -D warnings
    cargo fmt --check
    git ls-files '*.swift' | xargs swift format lint --strict

# Render gpui views to PNGs in apps/gui/tests/snapshots/ — local-only debug
# aid (gitignored), not a regression gate. Pixel diffs aren't bit-stable
# across machines, so committed regression checks live in tests/behavior.rs.
render-snapshots *args:
    cargo test -p eidola-gui --test visual {{ args }}

# Accept the current rendered output as the new local visual baseline.
render-snapshots-update:
    UPDATE_SNAPSHOTS=1 cargo test -p eidola-gui --test visual

# Run all tests (Rust + Swift on macOS)
test:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo test
    if [[ "$(uname -s)" == "Darwin" ]]; then
      echo "--- Swift tests (crates/eidola-app-core) ---"
      ( cd crates/eidola-app-core && swift test )
      echo "--- Swift tests (apps/macos) ---"
      ( cd apps/macos && swift test )
    fi

# Run integration tests (requires: just services && just db-reset)
test-integration:
    DATABASE_URL="${DATABASE_URL:-postgres://eidola@localhost/eidola}" CREDENTIAL_MASTER_KEY="${CREDENTIAL_MASTER_KEY:-0000000000000000000000000000000000000000000000000000000000000000}" cargo test -p eidola-server -- --ignored

# Run E2E webhook smoke tests (requires STRIPE_API_KEY)
test-webhook-smoke:
    ./scripts/test-webhook-smoke.sh

# --- Codegen ---

# Regenerate UniFFI Swift bindings
update-bindings:
    ./scripts/update-bindings.sh

# Regenerate OpenAPI spec
update-openapi:
    ./scripts/update-server-openapi.sh

# Rebuild XCFramework (dev, native arch only)
update-xcframework:
    ./scripts/update-xcframework-dev.sh

# Rebuild XCFramework (release, universal binary)
update-xcframework-release:
    ./scripts/update-xcframework.sh

# --- CI / Release ---

# Compute enclave measurements from tinfoil-config.yml and CVM artifacts
measure:
    ./scripts/artifact-manifest.sh measure

# Update artifact-manifest.json with current build digests and measurements
# Builds the OCI images plus the macOS app/CLI, then records their digests.
# Also stamps image digests into tinfoil-config.yml and computes enclave

# measurements. Requires macOS for the Nix-built app and CLI artifacts.
update-manifest:
    ./scripts/artifact-manifest.sh update --ensure-builder
