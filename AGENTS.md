# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Structure

```
eidolons/
├── crates/           # Rust crates
│   ├── eidolons-server/  # OpenAI-compatible AI proxy server
│   │   ├── src/
│   │   │   ├── main.rs       # HTTP server (axum + tokio), Config, serves OpenApiRouter
│   │   │   ├── lib.rs        # Module declarations, AppState (Clone via Arc), build_router()
│   │   │   ├── handlers.rs   # Core handlers: health, models, chat completions (axum extractors)
│   │   │   ├── helpers.rs    # Consolidated utilities: calendar, timestamp formatting, key period computation
│   │   │   ├── account.rs    # Account handlers (axum extractors)
│   │   │   ├── db.rs         # Database pool (deadpool-postgres) and query helpers
│   │   │   ├── stripe.rs     # Thin Stripe API client (checkout, subscriptions, portal)
│   │   │   ├── auth.rs       # TokenAuth + BasicAuth extractors, AnyValidator dispatch
│   │   │   ├── backend.rs    # ChatBackend trait and RedPill.ai implementation
│   │   │   ├── types.rs      # OpenAI API request/response types
│   │   │   ├── response.rs   # Eidolons response types with privacy metadata
│   │   │   ├── attestation.rs # RedPill TEE attestation signature fetching
│   │   │   ├── webhook.rs    # Stripe webhook signature verification and event dispatch
│   │   │   ├── credentials.rs # Credential issuance: key mgmt, encryption, GET /v1/keys, POST /v1/account/credentials
│   │   │   ├── error.rs      # ServerError enum, HTTP status mapping, IntoResponse impl
│   │   │   └── api_doc.rs    # OpenAPI schemas + security modifier (paths collected by OpenApiRouter)
│   │   ├── schema.sql        # PostgreSQL schema (billing, credential keys, nullifiers)
│   │   └── Containerfile     # StageX-based OCI build
│   └── eidolons-shared/  # Crux-based shared core (exclusive FFI generator)
│       ├── src/
│       │   ├── lib.rs        # FFI bridge (processEvent, handleResponse, view, capabilities)
│       │   ├── app.rs        # Crux App impl (Event, Model, ViewModel, Effect)
│       │   └── capabilities/ # Crux capabilities (e.g., eidolons)
│       ├── swift/            # Generated bindings (UniFFI + Crux types)
│       └── Package.swift     # Swift Package exposing EidolonsShared + SharedTypes
├── apps/
│   ├── cli/              # Pure Rust CLI app (clap + tokio)
│   │   ├── schema.sql        # Canonical SQLite schema (wallet, credentials)
│   │   └── src/
│   │       ├── main.rs       # CLI entrypoint: account, wallet, and credential commands
│   │       ├── config.rs     # Config file persistence (~/.config/eidolons/config.toml)
│   │       └── db.rs         # Embedded Turso/libSQL database with migrations
│   └── macos/            # macOS app (SwiftPM + Xcode wrapper)
│       ├── Sources/
│       │   ├── Eidolons/         # SwiftUI shell (Core.swift, ContentView.swift)
│       │   └── EidolonsEntrypoint/  # App entrypoint
│       ├── Xcode/            # Xcode project wrapper
│       ├── Support/          # Shared build files (Info.plist, scripts)
│       └── Package.swift     # Swift Package Manager config
├── docs/
│   └── design/           # Architecture Decision Records (ADRs)
├── tools/            # Build tooling
│   ├── uniffi-bindgen-swift/  # UniFFI binding generator
│   └── shared-typegen/        # Crux type generator for Swift
├── scripts/          # Dev and test scripts (wrapped by justfile)
├── justfile          # Task runner (primary development interface)
├── artifact-manifest.json    # Committed OCI image digests (reproducibility invariant)
├── compose.yaml              # Development environment (postgres + server)
├── docker-bake.hcl           # Reproducible OCI build settings (overlays compose.yaml)
├── .dockerignore             # Excludes non-build files from OCI build context
└── flake.nix         # Nix: CI checks, Swift/XCFramework builds
```

## Server Architecture

The server is an OpenAI-compatible proxy that translates requests to upstream AI providers. It includes a billing system with anonymous credentials for privacy-preserving usage tracking.

**Current upstream:** RedPill.ai (OpenAI-compatible, routes to various model providers)

**Database:** PostgreSQL 16+ (see `crates/eidolons-server/schema.sql`)

**Deployment:** Phala dstack — all services run inside a single Confidential VM (CVM) with encrypted disk backed by Intel TDX.

**CI:** Two workflows — `ci.yml` (self-hosted Mac: Nix checks, Swift builds/tests) and `oci.yml` (ubuntu-latest: OCI image builds, manifest verification, GHCR publishing).

**Image tagging:** `main` (rolling, updated on every merge), `v*` (immutable release tags), `sha-<short>` (per-commit). No `:latest`. Images published to `ghcr.io/<owner>/eidolons-server` and `ghcr.io/<owner>/eidolons-postgres`.

**Key design decisions:**
- Axum-based HTTP server with typed routing, extractors, and `utoipa-axum` OpenAPI integration
- Pure Rust TLS via `rustls-rustcrypto` (no C dependencies)
- Statically linked musl binaries for Linux deployment
- StageX-based OCI images (reproducible, `FROM scratch`, runs as non-root)
- Request-based (no sessions/caching in the proxy layer)
- Account auth (Basic + Argon2id) via `BasicAuth` extractor, chat completions auth via `TokenAuth` extractor
- Stripe integration via thin `reqwest` wrapper (no `async-stripe` dependency)

**API endpoints:** Defined in `crates/eidolons-server/openapi.json` (generated from utoipa annotations — see Conventions).

**Environment variables:**
- `REDPILL_API_KEY` (required) - RedPill API key
- `DATABASE_URL` (required) - PostgreSQL connection string
- `BIND_ADDR` (default: `127.0.0.1:8080`) - Address to bind
- `STRIPE_API_KEY` (optional) - Stripe secret key; account billing endpoints return 503 without it
- `STRIPE_WEBHOOK_SECRET` (optional) - Stripe webhook signing secret; webhook endpoint returns 503 without it
- `CREDENTIAL_MASTER_KEY` (optional) - Hex-encoded 32-byte AES-256 master key for issuer key encryption; credential issuance endpoints return 503 without it

## Crux Architecture

The apps use [Crux](https://redbadger.github.io/crux/) for cross-platform state management. The architecture separates the core (Rust) from the shells (Swift for macOS, Rust for CLI):

```
┌─────────────────────────────────────────────────────────┐
│  Swift Shell (apps/macos)                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Core.swift - handles event/effect loop           │  │
│  │  - Sends Events to core via processEvent()        │  │
│  │  - Handles Effects (Render, Eidolons capability)  │  │
│  │  - Updates ViewModel for SwiftUI                  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │ FFI (UniFFI + bincode)
┌─────────────────────▼───────────────────────────────────┐
│  Crux Core (crates/eidolons-shared)                      │
│  - Event: user actions (e.g., Greet)                    │
│  - Model: private app state                             │
│  - ViewModel: public view state                         │
│  - Effect: side-effects for shell to handle             │
│  - Capabilities: Render, Eidolons (calls capability impls) │
└─────────────────────────────────────────────────────────┘
```

**Key pattern:** The core never performs side-effects directly. It emits Effects that the shell handles, then the shell sends responses back via `handleResponse()`.

**Capability implementations:** Pure Rust crates in `crates/` (e.g., `eidolons-perception`) implement capability logic. These are compiled into `eidolons-shared` and exposed via UniFFI, so the Swift shell can call them directly.

**Two codegen pipelines:**
- `uniffi-bindgen-swift` → FFI bridge (`processEvent`, `handleResponse`, `view`)
- `crux_core::typegen` → Domain types (`Event`, `Effect`, `ViewModel`) with bincode serialization

## CLI Database & Migrations

The CLI uses an embedded [Turso](https://crates.io/crates/turso) (pure-Rust libSQL) database at `~/Library/Application Support/eidolons/eidolons.db` for local app data (wallet credentials, conversation history, etc.).

**Schema management:**
- `apps/cli/schema.sql` is the canonical schema — always reflects the current desired state
- Fresh installs apply `schema.sql` directly via `execute_batch` and set `PRAGMA user_version` to `LATEST_VERSION`
- Existing databases run incremental migrations in `db.rs` (gated by `user_version`)

**Adding a migration:**
1. Update `schema.sql` to the new desired state
2. Add a `MIGRATION_N` constant in `db.rs` with the ALTER/CREATE statements
3. Add an `if current_version < N` block in `migrate()` that runs the migration and sets `user_version`
4. Bump `LATEST_VERSION`
5. Run `cargo test -p eidolons-cli` — the `migrations_match_schema` test structurally compares a fresh-from-schema database against a fully-migrated database (via `PRAGMA table_info`, `PRAGMA index_info`, and view SQL)

**Limitations:** The `turso` crate does not support `ALTER TABLE ALTER COLUMN` (a libSQL C extension). To add `NOT NULL` columns, use `ADD COLUMN ... DEFAULT <value>` — the default persists and must also be declared in `schema.sql` so both paths match.

## Build Commands

**Prerequisites:** `rustup`, `just`, `docker`

The `justfile` is the primary development interface. Run `just` to see all available recipes.

```bash
# --- Development (daily workflow) ---

just db                       # Start postgres in Docker
just db-reset                 # Apply/reset schema.sql
cargo run -p eidolons-server  # Run server on host (fast iteration)
cargo run -p eidolons-cli     # Run CLI app (iocraft TUI shell)
cargo test                    # Run tests
just check                    # Lint (clippy + fmt)

# Full stack in containers (validates OCI build + service topology)
just dev

# Full stack with Stripe webhook forwarding (requires STRIPE_API_KEY)
just dev-stripe

# --- Testing ---

just test-integration         # Integration tests (requires: just db && just db-reset)
just test-webhook-smoke       # E2E webhook smoke tests (requires STRIPE_API_KEY)

# --- Codegen (after changing Rust APIs/types) ---

just update-bindings          # UniFFI Swift bindings + Crux types
just update-openapi           # OpenAPI spec
just update-xcframework       # XCFramework (dev, native arch only)

# --- OCI ---

just oci-build                # Build all OCI images (reproducible, via buildx bake)
just update-manifest          # Rebuild images and update artifact-manifest.json

# --- CI / Release (Nix) ---

just ci-check                 # nix flake check (fmt, clippy, tests, artifact freshness)
just ci-build-xcframework     # XCFramework via Nix (universal binary)

# --- Nix-based codegen (for CI parity) ---

nix run '.#update-eidolons-shared-swift-bindings'
nix run '.#update-server-openapi'
```

## Key Files

| File | Purpose |
|------|---------|
| `justfile` | Task runner — primary development interface |
| `compose.yaml` | Dev environment: postgres + server + stripe-cli (test profile) |
| `docker-bake.hcl` | Reproducible OCI build settings (overlays compose.yaml) |
| `artifact-manifest.json` | Committed OCI image digests — CI verifies builds match |
| `crates/eidolons-server/src/helpers.rs` | Consolidated calendar, timestamp formatting, key period utilities |
| `crates/eidolons-server/Containerfile` | StageX-based OCI image build |
| `crates/eidolons-server/schema.sql` | PostgreSQL schema (billing, credentials, nullifiers) |
| `.env.example` | Template for local environment variables |
| `flake.nix` | Nix: CI checks, Swift codegen, XCFramework builds |
| `rust-toolchain.toml` | Pinned Rust version and targets |
| `Cargo.toml` | Workspace config, release profile (LTO, single codegen unit) |
| `crates/eidolons-server/Cargo.toml` | Server dependencies |
| `crates/eidolons-shared/Package.swift` | Shared core Swift Package (EidolonsShared + SharedTypes) |
| `crates/eidolons-shared/src/lib.rs` | FFI bridge + capability re-exports |
| `crates/eidolons-shared/src/app.rs` | Crux App implementation (Event, Model, ViewModel, Effect) |
| `apps/cli/schema.sql` | Canonical CLI SQLite schema (used for fresh installs) |
| `apps/cli/src/db.rs` | Embedded Turso database: open, initialize, migrate |
| `apps/cli/src/main.rs` | CLI entrypoint: account, wallet, and credential commands |
| `apps/macos/Package.swift` | macOS app Swift Package config |
| `apps/macos/Sources/Eidolons/Core.swift` | Swift shell bridge (handles Crux event/effect loop) |
| `scripts/dev-stripe.sh` | Start full stack with Stripe webhook forwarding |
| `scripts/test-webhook-smoke.sh` | E2E webhook smoke tests |
| `apps/macos/Support/package-app.sh` | CLI build script for .app bundle |

## Design Documents

Architecture decisions are recorded in [`docs/design/`](docs/design/). See the
[index](docs/design/README.md) for a full list. Key decisions:

- [Model Weight Management](docs/design/model-weight-management.md) — weights as pinned dependencies, hash-verified at every boundary
- [Pure Rust, Zero C Dependencies](docs/design/pure-rust-zero-c-dependencies.md) — rustls-rustcrypto, webpki-roots, no C cross-compiler needed
- [Reproducible Builds](docs/design/reproducible-builds.md) — Nix, Crane, deterministic settings, CI-verified generated artifacts
- [Crux Cross-Platform Architecture](docs/design/crux-cross-platform-architecture.md) — Elm-like core/shell split, UniFFI, bincode FFI bridge
- [OpenAI-Compatible Proxy Server](docs/design/openai-compatible-proxy-server.md) — canonical API format, stateless proxy, distroless OCI
- [On-Device Inference with Burn](docs/design/on-device-inference-with-burn.md) — pure Rust ML, WGPU GPU backend, model-per-crate

## Conventions

- Pure Rust dependencies preferred (for cross-compilation)
- `just` is the task runner — wrap scripts and common commands as recipes
- `compose.yaml` defines the dev environment; `docker-bake.hcl` overlays reproducible build settings
- `docker buildx bake` is the single entry point for all OCI image builds
- Server OCI images are built with StageX (reproducible, `FROM scratch`, runs as non-root)
- Nix is used for CI quality gates and Swift/XCFramework builds, not daily Rust development
- `rustup` + `rust-toolchain.toml` manages the Rust toolchain for development
- OpenAI API format as the canonical interface
- Server API is documented via utoipa `#[utoipa::path]` annotations on handler functions and `ToSchema` derives on request/response types. `OpenApiRouter` (in `lib.rs::build_router()`) collects paths and recursively discovers schemas automatically — only SSE streaming types that aren't referenced from path annotations are listed manually in `api_doc.rs`. When adding or changing server endpoints, add the annotation on the handler and register it in `build_router()` via `routes!()`, then run `just update-openapi` to regenerate the committed `openapi.json`
- Deterministic builds (no timestamps, fixed codegen)
- `artifact-manifest.json` records expected OCI digests; CI verifies builds match and suggests updates on PRs
- Before committing, ensure `README.md` and `AGENTS.md` are updated to reflect any changes (new files, endpoints, env vars, build commands, etc.)
- Omit any tool-specific "co-authored by" lines from commit messages
