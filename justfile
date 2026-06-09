set dotenv-load

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
    docker compose exec postgres psql -U eidola -d eidola -v ON_ERROR_STOP=1 -c "ALTER ROLE eidola IN DATABASE eidola SET search_path = public" -c "SET search_path TO public" -f /docker-entrypoint-initdb.d/schema.sql

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
          open -W "crates/eidola-gui/build/Eidola.app" --args {{ args }}
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
    rumdl check .

# Apply auto-fixable markdown formatting via rumdl
format:
    cargo fmt
    rumdl check . --fix

# Render gpui views to PNGs in crates/eidola-gui/tests/snapshots/ — local-only debug
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

# List attestant signing keys on a PKCS#11 token (YubiKey/SmartCard/HSM)
# and print PIN-free cosign --key URIs (no pin-value, no slot-id). Reads
# only public objects, so it never prompts for or emits a PIN. Copy the
# printed uri into EIDOLA_ATTESTANT_COSIGN_KEY. Optional --module-path
# /path/to/libykcs11.{dylib,so} if auto-detection misses it. Full runbook:
# docs/contributing/release-attestant-yubikey.md
release-list-keys *args:
    cargo run -q -p release-tool -- pkcs11 list {{ args }}

# Capture INFORMATIONAL hardware-provenance evidence for an attestant key
# (YubiKey-PIV): runs ykman to write the slot-9c attestation cert + F9
# intermediate under releases/trust/attestant-provenance/<id>/ and fills
# meta.json (serial/firmware/PIN+touch policy parsed from the attestation
# cert). Not consulted by trust evaluation — see that dir's README.
# Non-YubiKey attestants populate the bundle by hand.
release-provenance-capture id *args:
    cargo run -q -p release-tool -- provenance capture --attestant-id {{ id }} {{ args }}

# (Re)derive meta.json device fields from a bundle's committed attestation
# cert — offline, no device/ykman. No args enriches every bundle; pass
# --attestant-id <id> for one.
release-provenance-enrich *args:
    cargo run -q -p release-tool -- provenance enrich {{ args }}

# Verify each committed provenance bundle's attestation cert matches the
# fingerprint its meta.json claims and is still pinned (vendor-neutral;
# CI-friendly). A bundle for an unpinned fingerprint fails as stale.
release-provenance-check *args:
    cargo run -q -p release-tool -- provenance check {{ args }}

# Verify a tag that CI has already built+signed. Fetches the signed manifest
# from the GitHub release, verifies the Sigstore bundle against the embedded
# trust root, compares against the committed manifest, and shows the diff
# against the prior release for human review. Run before `release-attest`.
release-verify tag:
    cargo run -q -p release-tool -- verify {{ tag }}

# Interactive: render each claim, prompt to type 'yes' to affirm, sign
# with cosign (local PEM, PKCS#11 URI for a YubiKey/SmartCard, or any
# cosign-supported KMS URI), upload attestation + release.json, mark
# release as latest. Reads attestant identity from
# EIDOLA_ATTESTANT_COSIGN_KEY / _ID / _NAME / _JURISDICTION env vars
# (preferred — set once in your shell profile or .envrc). For local
# PEM keys, also set COSIGN_PASSWORD (empty string is OK for passphrase-
# less throwaway keys). For a YubiKey-PIV key, get a PIN-free URI from
# `just release-list-keys` and set it as EIDOLA_ATTESTANT_COSIGN_KEY; the
# PIN is prompted for automatically when COSIGN_PKCS11_PIN is unset (never
# put the PIN in the URI — cosign's own prompt fails against a YubiKey, so
# release-tool supplies it via the env var for the cosign child processes).
# Trailing args are forwarded to `release-tool
# attest`, so per-invocation overrides also work, e.g.
#   just release-attest v0.0.8 --cosign-key /path/to/cosign.key \
#       --attestant-id mike-marcacci \
#       --attestant-name "Michael Marcacci" \
#       --jurisdiction "the State of California, United States"
#
# PKCS#11 note: cosign's PKCS#11 support is a build-time option, not a
# separate binary — the default Homebrew and GitHub-release `cosign`
# binaries are built without it and fail any PKCS#11 op with "This cosign
# was not built with pkcs11-tool support!". For a YubiKey-PIV key
# (`pkcs11:slot-id=…`), build cosign from source with the pivkey/pkcs11key
# tags (cgo, so a C toolchain is required):
#   CGO_ENABLED=1 go install -tags=pivkey,pkcs11key \
#       github.com/sigstore/cosign/v2/cmd/cosign@latest
# (this is the `go install` equivalent of cosign's `cosign-pivkey-pkcs11key`
# Makefile target) and ensure $(go env GOPATH)/bin precedes Homebrew on
# PATH. Alternatively use a KMS URI instead: KMS-backed keys (`awskms:`,
# `gcpkms:`, `azurekms:`, `hashivault:`) work with the stock cosign binary.
# Full provisioning runbook: docs/contributing/release-attestant-yubikey.md
release-attest tag *args:
    cargo run -q -p release-tool -- attest {{ tag }} {{ args }}
