# AGENTS.md

Guidance for AI coding agents working in this repository.

## Server Architecture

The server is an OpenAI-compatible proxy that translates requests to upstream AI providers. It includes a billing system with anonymous credentials for privacy-preserving usage tracking.

**Current upstream:** RedPill.ai (OpenAI-compatible, routes to various model providers)

**Database:** PostgreSQL 17+ (see `crates/eidolons-server/schema/schema.sql`)

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

The macOS app uses [Crux](https://redbadger.github.io/crux/) for cross-platform state management. The CLI (`apps/cli/`) is a standalone clap/tokio app and does not use Crux.

**Key pattern:** The core never performs side-effects directly. It emits Effects that the shell handles, then the shell sends responses back via `handleResponse()`.

**Capability implementations:** Pure Rust crates in `crates/` implement capability logic. These are compiled into `eidolons-shared` and exposed via UniFFI, so the Swift shell can call them directly.

**Two codegen pipelines:**
- `uniffi-bindgen-swift` → FFI bridge (`processEvent`, `handleResponse`, `view`)
- `crux_core::typegen` → Domain types (`Event`, `Effect`, `ViewModel`) with bincode serialization

## CLI Database & Migrations

The CLI uses an embedded [Turso](https://crates.io/crates/turso) (pure-Rust libSQL) database at `~/Library/Application Support/eidolons/eidolons.db` for local app data (wallet credentials, conversation history, etc.).

**Schema management:**
- `apps/cli/schema/schema.sql` is the canonical schema — always reflects the current desired state
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

## Conventions

- Pure Rust dependencies preferred (for cross-compilation)
- `just` is the task runner — wrap scripts and common commands as recipes
- Server OCI images are built with StageX (reproducible, `FROM scratch`, runs as non-root)
- Nix is used for CI quality gates and Swift/XCFramework builds, not daily Rust development
- `rustup` + `rust-toolchain.toml` manages the Rust toolchain for development
- OpenAI API format as the canonical interface
- Server API is documented via utoipa `#[utoipa::path]` annotations on handler functions and `ToSchema` derives on request/response types. `OpenApiRouter` (in `lib.rs::build_router()`) collects paths and recursively discovers schemas automatically — only SSE streaming types that aren't referenced from path annotations are listed manually in `api_doc.rs`. When adding or changing server endpoints, add the annotation on the handler and register it in `build_router()` via `routes!()`, then run `just update-openapi` to regenerate the committed `openapi.json`
- `artifact-manifest.json` records expected OCI digests; CI verifies builds match and suggests updates on PRs
- Before committing, ensure `README.md` and `AGENTS.md` are updated to reflect any changes (new files, endpoints, env vars, build commands, etc.)
- Omit any tool-specific "co-authored by" lines from commit messages
