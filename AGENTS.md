# AGENTS.md

Guidance for AI coding agents working in this repository.

## Server Architecture

The server is an OpenAI-compatible proxy that translates requests to upstream AI providers. It includes a billing system with anonymous credentials for privacy-preserving usage tracking.

**Current upstream:** Tinfoil (inference.tinfoil.sh) — OpenAI-compatible, all models run in confidential enclaves (AMD SEV-SNP / Intel TDX / NVIDIA CC)

**Database:** PostgreSQL 17+ (see `crates/eidolons-server/schema/schema.sql`)

**Deployment:** Tinfoil Containers — all services run inside confidential enclaves (AMD SEV-SNP). The Tinfoil shim handles TLS termination with attestation-bearing certificates; the server runs plain HTTP behind it.

**CI:** Two workflows — `ci.yml` (self-hosted Mac: Nix checks, Swift builds/tests) and `oci.yml` (ubuntu-latest: OCI image builds, manifest verification, GHCR publishing).

**Image tagging:** `main` (rolling, updated on every merge), `v*` (immutable release tags), `sha-<short>` (per-commit). No `:latest`. Images published to `ghcr.io/<owner>/eidolons-server` and `ghcr.io/<owner>/eidolons-postgres`.

**Key design decisions:**
- Axum-based HTTP server with typed routing, extractors, and `utoipa-axum` OpenAPI integration
- Plain HTTP internally; TLS terminated by Tinfoil Container shim with attestation-bearing certificates (attestation hash + HPKE key encoded in SANs, issued by public CA)
- Tinfoil attestation verification via `tinfoil-verifier` crate — verifies SEV-SNP hardware attestation per-connection, caching verified fingerprints for fast reconnections; handles load-balanced deployments
- Statically linked musl binaries for Linux deployment
- StageX-based OCI images (reproducible, `FROM scratch`, runs as non-root)
- Request-based (no sessions/caching in the proxy layer)
- Account auth (Basic + Argon2id) via `BasicAuth` extractor, chat completions auth via `TokenAuth` extractor
- Stripe integration via thin `reqwest` wrapper (no `async-stripe` dependency)

**API endpoints:** Defined in `crates/eidolons-server/openapi.json` (generated from utoipa annotations — see Conventions).

**Environment variables:**
- `TINFOIL_API_KEY` (required) - Tinfoil inference API key
- `DATABASE_URL` (required) - PostgreSQL connection string
- `CREDENTIAL_MASTER_KEY` (required) - Hex-encoded 32-byte AES-256 key for encrypting issuer private keys at rest in Postgres; in production, injected as a Tinfoil secret; in local dev, use the all-zeros key from `.env.example`
- `BIND_ADDR` (default: `127.0.0.1:8443`) - Address to bind (HTTP); Containerfile overrides to `0.0.0.0:8080`
- `STRIPE_API_KEY` (optional) - Stripe secret key; account billing endpoints return 503 without it
- `STRIPE_WEBHOOK_SECRET` (optional) - Stripe webhook signing secret; webhook endpoint returns 503 without it
- `TINFOIL_BASE_URL` (optional) - Override the default Tinfoil API base URL (`https://inference.tinfoil.sh/v1`)
- `TINFOIL_PRICING_OVERRIDES` (optional) - JSON object overriding per-model pricing; e.g. `{"kimi-k2-5":{"input":2.0,"output":6.0}}`. Token-based models accept `input`/`output` ($/M tokens); per-request models accept `request` ($/request). See `backend.rs` `MODEL_CATALOG` for defaults
- `PRICING_MARKUP` (optional) - Pricing markup factor applied to all model prices (default: `1.5`)

**Tinfoil Containers / TEE integration:**

The server runs as plain HTTP inside a Tinfoil Container. The Tinfoil shim handles TLS termination externally, generating attestation-bearing certificates (attestation hash and HPKE key encoded in SANs, issued by a public CA via ACME). The shim serves `/.well-known/tinfoil-attestation` for client-side verification.

The credential master key (`CREDENTIAL_MASTER_KEY`, AES-256, hex-encoded) encrypts issuer private keys at rest in Postgres. In production, it is injected as a Tinfoil secret (environment variable, encrypted, only accessible inside the enclave). In local dev, use the all-zeros dev key from `.env.example`. The key must remain stable across upgrades so encrypted issuer keys in the database remain accessible.

The container has access to `/dev/sev-guest` (via the undocumented `devices` field in `tinfoil-config.yml`) for requesting SEV-SNP attestation reports. The pre-generated attestation document and TLS key material are also available at `/tinfoil/` inside the container.

**Tinfoil attestation verification (`crates/tinfoil-verifier/`):**

On startup the server verifies the Tinfoil inference enclave's hardware attestation before sending any traffic. The `tinfoil-verifier` crate handles this via `attesting_client()`:

1. Fetches the attestation bundle from the Tinfoil ATC service for initial bootstrap verification
2. Verifies the AMD VCEK certificate chain (embedded Genoa ARK → ASK → VCEK) using RSA-PSS(SHA-384)
3. Verifies the SEV-SNP attestation report's ECDSA-P384 signature against the VCEK public key
4. Validates TCB policy (minimum firmware versions: bl≥0x07, snp≥0x0e, ucode≥0x48)
5. Checks the report measurement against `ALLOWED_MEASUREMENTS` in `measurements.rs`
6. Cross-checks the enclave TLS certificate's SPKI fingerprint against `report_data[0..32]`
7. Returns a `reqwest::Client` with a custom `ServerCertVerifier` that verifies attestation per-connection

Unlike static cert pinning, verified SPKI fingerprints are cached so reconnections to already-attested instances are instant. New connections fetch `/.well-known/tinfoil-attestation` from the connected instance, verify the SEV-SNP report, and cache the result. VCEKs are cached per chip ID and fetched from AMD KDS when encountering a new chip. This handles load-balanced deployments where each instance has its own TLS key. Pure Rust dependencies (`sev`, `x509-cert`, `der`) — no OpenSSL.

**Compose files:**
- `compose.yaml` — local development: postgres + server + stripe-cli

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
