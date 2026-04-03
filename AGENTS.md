# AGENTS.md

Guidance for AI coding agents working in this repository.

## Server Architecture

The server is an OpenAI-compatible proxy that translates requests to upstream AI providers. It includes a billing system with anonymous credentials for privacy-preserving usage tracking.

**Current upstream:** Tinfoil (inference.tinfoil.sh) — OpenAI-compatible, all models run in confidential enclaves (AMD SEV-SNP / Intel TDX / NVIDIA CC)

**Database:** PostgreSQL 17+ (see `crates/eidola-server/schema/schema.sql`)

**Deployment:** Tinfoil Containers — all services run inside confidential enclaves (AMD SEV-SNP). The Tinfoil shim handles TLS termination with attestation-bearing certificates; the server runs plain HTTP behind it.

**CI:** Single `ci.yml` workflow with four jobs — `rust-checks` (ubuntu-24.04: cargo fmt/clippy/test, OpenAPI freshness), `oci` (ubuntu-24.04: OCI image builds, OCI subset verification, GHCR publishing), `apple` (self-hosted Mac: Swift formatting/bindings freshness, Nix-based macOS app and CLI universal binary builds, Swift tests), and `artifact-manifest` (ubuntu-24.04: merges the OCI and macOS artifact digests, recomputes enclave measurements from `tinfoil-config.yml` + CVM artifacts, and verifies the full committed manifest). The `oci` and `apple` jobs gate on `rust-checks` to avoid wasting resources on failing PRs.

**Image tagging:** `main` (rolling, updated on every merge), `v*` (immutable release tags), `sha-<short>` (per-commit). No `:latest`. Images published to `ghcr.io/<owner>/eidola-server`, `ghcr.io/<owner>/eidola-cli`, and `ghcr.io/<owner>/eidola-postgres`.

**Key design decisions:**
- Axum-based HTTP server with typed routing, extractors, and `utoipa-axum` OpenAPI integration
- Plain HTTP internally; TLS terminated by Tinfoil Container shim with attestation-bearing certificates (attestation hash + HPKE key encoded in SANs, issued by public CA)
- Tinfoil attestation verification via `tinfoil-verifier` crate — verifies SEV-SNP hardware attestation per-connection, caching verified fingerprints for fast reconnections; handles load-balanced deployments
- Deterministic enclave measurement via `measure-enclave` crate — pre-computes SEV-SNP and TDX measurements from source, committed in `artifact-manifest.json`
- Statically linked musl binaries for Linux deployment
- StageX-based OCI images (reproducible, `FROM scratch`, runs as non-root)
- Request-based (no sessions/caching in the proxy layer)
- Account auth (Basic + Argon2id) via `BasicAuth` extractor, chat completions auth via `TokenAuth` extractor
- Stripe integration via thin `reqwest` wrapper (no `async-stripe` dependency)

**API endpoints:** Defined in `crates/eidola-server/openapi.json` (generated from utoipa annotations — see Conventions).

**Environment variables:**
- `TINFOIL_API_KEY` (required) - Tinfoil inference API key
- `DATABASE_URL` (required) - PostgreSQL connection string
- `DATABASE_PASSWORD` (optional) - PostgreSQL password; in production with an external database, inject this as a Tinfoil secret instead of embedding it in `DATABASE_URL`
- `DATABASE_SSL_CERT` (optional) - PEM-encoded root CA certificate for PostgreSQL TLS verification; public material, so this can be passed as a normal env var
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

When using an external PostgreSQL instance until Tinfoil supports persistent disks, keep connection metadata in `DATABASE_URL`, inject `DATABASE_PASSWORD` as a Tinfoil secret, and pass the PEM-encoded server root CA in `DATABASE_SSL_CERT` if the server certificate does not chain to a default WebPKI root. The root CA certificate is public material; only the password and private keys need to be kept secret.

The container has access to `/dev/sev-guest` (via the undocumented `devices` field in `tinfoil-config.yml`) for requesting SEV-SNP attestation reports. The pre-generated attestation document and TLS key material are also available at `/tinfoil/` inside the container.

**Enclave measurement (`crates/measure-enclave/`):**

The `measure-enclave` binary pre-computes the hardware attestation measurements that a legitimate Tinfoil Container will produce. The measurement is a deterministic function of:

1. OVMF firmware (pinned from `tinfoilsh/edk2`)
2. CVM kernel + initrd (versioned from `tinfoilsh/cvmimage`, hash-verified)
3. Kernel command line (embeds dm-verity roothash + SHA-256 of `tinfoil-config.yml`)
4. vCPU count and type

The binary uses `sev` (with `crypto_nossl` feature — pure Rust, no OpenSSL) for SEV-SNP launch digest computation and `tdx-measure` for TDX RTMR1/RTMR2 runtime measurements. Both work natively on macOS.

`tinfoil-config.yml` is the Tinfoil Container configuration. It references container images by digest (from `artifact-manifest.json`), declares `_HASH` env vars for measured secrets (Argon2id hashes generated via `cargo run -p hash-secret`), and specifies CVM resources (cpus, memory). The SHA-256 of this file is embedded in the kernel command line and bound into the enclave measurement, so any change to the config produces a different measurement.

The measurement flow: `source → deterministic OCI build → digest → tinfoil-config.yml (with digest) → cmdline (with config hash) → measurement`. All values are committed in `artifact-manifest.json` and verified by CI. CVM artifacts are cached locally at `~/.cache/eidola/cvm/`.

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

## App Core Architecture

The macOS app and CLI share a common Rust core (`crates/eidola-app-core/`) exposed to Swift via direct [UniFFI](https://mozilla.github.io/uniffi-rs/) bindings. Rust functions and types are exported with `#[uniffi::export]`, `#[derive(uniffi::Object)]`, `#[derive(uniffi::Record)]`, and `#[derive(uniffi::Enum)]`. Async operations use `#[uniffi::export(async)]` to bridge Rust futures to Swift async/await. No serialization layer, event/effect pattern, or Crux dependency — Swift calls Rust functions directly and gets native Swift types back.

**Crate layout:** Pure Rust crates in `crates/` implement capability logic. The `crates/` tree also contains the Rust code generation binary (`uniffi-bindgen-swift`) plus operational utilities such as `generate-openapi`, `tinfoil-shim-mock`, `hash-secret`, and `measure-enclave`.

**Codegen pipeline:**
- `uniffi-bindgen-swift` (workspace crate under `crates/`) → FFI bridge (Swift bindings + C headers)

## CLI Database & Migrations

The CLI uses an embedded [Turso](https://crates.io/crates/turso) (pure-Rust libSQL) database at `~/Library/Application Support/eidola/eidola.db` for local app data (wallet credentials, conversation history, etc.).

**Schema management:**
- `apps/cli/schema/schema.sql` is the canonical schema — always reflects the current desired state
- Fresh installs apply `schema.sql` directly via `execute_batch` and set `PRAGMA user_version` to `LATEST_VERSION`
- Existing databases run incremental migrations in `db.rs` (gated by `user_version`)

**Adding a migration:**
1. Update `schema.sql` to the new desired state
2. Add a `MIGRATION_N` constant in `db.rs` with the ALTER/CREATE statements
3. Add an `if current_version < N` block in `migrate()` that runs the migration and sets `user_version`
4. Bump `LATEST_VERSION`
5. Run `cargo test -p eidola-cli` — the `migrations_match_schema` test structurally compares a fresh-from-schema database against a fully-migrated database (via `PRAGMA table_info`, `PRAGMA index_info`, and view SQL)

**Limitations:** The `turso` crate does not support `ALTER TABLE ALTER COLUMN` (a libSQL C extension). To add `NOT NULL` columns, use `ADD COLUMN ... DEFAULT <value>` — the default persists and must also be declared in `schema.sql` so both paths match.

## Build Commands

**Prerequisites:** `rustup`, `just`, `docker`

The `justfile` is the primary development interface. Run `just` to see all available recipes.

## Conventions

- Pure Rust dependencies preferred (for cross-compilation)
- Keep Rust workspace packages under `crates/`; do not add a separate top-level `tools/` tree
- `just` is the task runner — wrap scripts and common commands as recipes
- Server and CLI OCI images are built with StageX (reproducible, `FROM scratch`, runs as non-root)
- Nix is used for CI quality gates and Swift/XCFramework builds, not daily Rust development
- `rustup` + `rust-toolchain.toml` manages the Rust toolchain for development
- OpenAI API format as the canonical interface
- Server API is documented via utoipa `#[utoipa::path]` annotations on handler functions and `ToSchema` derives on request/response types. `OpenApiRouter` (in `lib.rs::build_router()`) collects paths and recursively discovers schemas automatically — only SSE streaming types that aren't referenced from path annotations are listed manually in `api_doc.rs`. When adding or changing server endpoints, add the annotation on the handler and register it in `build_router()` via `routes!()`, then run `just update-openapi` to regenerate the committed `openapi.json`
- `artifact-manifest.json` (v1 format) records expected OCI digests, macOS app/CLI Nix `narHash` values, and enclave measurements (SEV-SNP + TDX) with type/platform metadata; CI verifies the full file by merging digests captured from the real OCI and macOS build jobs and recomputing enclave measurements from `tinfoil-config.yml`. Use `just update-manifest` to regenerate it on macOS with the pinned amd64 BuildKit builder plus the local Nix macOS builds
- Before committing, ensure `README.md` and `AGENTS.md` are updated to reflect any changes (new files, endpoints, env vars, build commands, etc.)
- Omit any tool-specific "co-authored by" lines from commit messages
