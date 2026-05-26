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

# Build a system: server, cli, or gui
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
        # On macOS, also assemble a proper .app bundle. AppKit needs an
        # Info.plist alongside the binary to treat the process as a real
        # app (vs. a command-line tool); otherwise menu key-equivalent
        # dispatch breaks when no window has key focus.
        if [[ "$(uname -s)" == "Darwin" ]]; then
          ./scripts/package-gui-app.sh debug
        fi
        ;;
      *)
        echo "error: unknown system '{{ system }}' (expected: server, cli, gui)" >&2
        exit 1
        ;;
    esac

# Build and run a system: server, cli, or gui
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
        # Build + assemble the .app, then `open` it so AppKit launches us
        # in app-mode (full menu / key-equivalent dispatch). `cargo run`
        # alone would launch the bare binary, which AppKit treats as a
        # tool — visible in the menu bar showing "eidola-gui" instead of
        # "Eidola" and breaking ⌘N / ⌘Q etc. when no window has key
        # focus. On non-macOS we fall back to `cargo run`.
        if [[ "$(uname -s)" == "Darwin" ]]; then
          just build gui
          open -W "apps/gui/build/Eidola.app" --args {{ args }}
        else
          cargo run -p eidola-gui -- {{ args }}
        fi
        ;;
      *)
        echo "error: unknown system '{{ system }}' (expected: server, cli, gui)" >&2
        exit 1
        ;;
    esac

# --- Checks & Tests (inner loop, runs on host) ---

# Lint and format check
check:
    cargo clippy --all-targets -- -D warnings
    cargo fmt --check

# Render gpui views to PNGs in apps/gui/tests/snapshots/ — local-only debug
# aid (gitignored), not a regression gate. Pixel diffs aren't bit-stable
# across machines, so committed regression checks live in tests/behavior.rs.
render-snapshots *args:
    cargo test -p eidola-gui --test visual {{ args }}

# Accept the current rendered output as the new local visual baseline.
render-snapshots-update:
    UPDATE_SNAPSHOTS=1 cargo test -p eidola-gui --test visual

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

# Regenerate OpenAPI spec
update-openapi:
    ./scripts/update-server-openapi.sh

# --- CI / Release ---

# Compute enclave measurements from tinfoil-config.yml and CVM artifacts
measure:
    ./scripts/artifact-manifest.sh measure

# Update artifact-manifest.json with current build digests and measurements.
# Builds the OCI images plus the CLI macOS universal binary, then records
# their digests. Also stamps image digests into tinfoil-config.yml and
# computes enclave measurements. Requires macOS for the Nix-built CLI.
update-manifest:
    ./scripts/artifact-manifest.sh update --ensure-builder

# Verify a tag that CI has already built+signed. Fetches the signed manifest
# from the GitHub release, verifies the Sigstore bundle against the embedded
# trust root, compares against the committed manifest, and shows the diff
# against the prior release for human review. Run before `release-attest`.
release-verify tag:
    cargo run -q -p release-tool -- verify {{ tag }}

# Interactive: render each claim, prompt to type 'yes' to affirm, sign
# with the configured hardware key, upload attestation + release.json,
# mark release as latest. Reads attestant identity from EIDOLA_ATTESTANT_*
# env vars; cosign key from EIDOLA_ATTESTANT_KEY (typically a PKCS#11 URI).
release-attest tag:
    cargo run -q -p release-tool -- attest {{ tag }}
