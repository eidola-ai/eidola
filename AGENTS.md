# AGENTS.md

Guidance for AI coding agents working in this repository.

## Server Architecture

The server is an OpenAI-compatible proxy that translates requests to upstream AI providers. It includes a billing system with anonymous credentials for privacy-preserving usage tracking.

**Current upstream:** Tinfoil (inference.tinfoil.sh) — OpenAI-compatible, all models run in confidential enclaves (AMD SEV-SNP / Intel TDX / NVIDIA CC)

**Database:** PostgreSQL 17+ (see `crates/eidola-server/schema/schema.sql`)

**Deployment:** Tinfoil Containers — all services run inside confidential enclaves (AMD SEV-SNP). The Tinfoil shim handles TLS termination with attestation-bearing certificates; the server runs plain HTTP behind it.

**CI:** `ci.yml` contains four jobs — `rust-checks` (ubuntu-24.04: cargo fmt/clippy/test, OpenAPI freshness), `oci` (ubuntu-24.04), `apple` (self-hosted Mac: Nix-based macOS universal-binary builds for both the cli and the GUI .app bundle), and `artifact-manifest` (ubuntu-24.04). The `oci` job mirrors the local `just update-manifest` flow end-to-end in five inline phases — (1) bake server + postgres, (2) stamp the freshly-built server digest into the workspace `tinfoil-config.yml`, (3) recompute the enclave from the stamped config and overwrite `releases/trust/server-enclave.json`, (4) bake cli (its build context COPYs the just-overwritten `server-enclave.json`), (5) emit the stamped config, recomputed enclave, and combined OCI partial as job outputs — and verifies the built OCI subset against the committed `artifact-manifest.json`. `apple` `needs: oci`, installs the stamped config + recomputed enclave from `oci`'s outputs, then builds the macOS universal CLI against that same trust root. `artifact-manifest` gates on `rust-checks`, `oci`, and `apple`; it materializes all three `oci` outputs into the workspace, runs `verify-full` (which recomputes the enclave from the stamped config and compares the composed manifest to the committed `artifact-manifest.json`), and on PRs posts a REQUEST_CHANGES review whose body contains all three files (`tinfoil-config.yml`, `releases/trust/server-enclave.json`, `artifact-manifest.json`) verbatim. Because CI executes the entire generation chain (server build → stamp → measure → cli build → compose) inside one job rather than verifying disjoint partials, those three suggestions are guaranteed internally consistent: committing them as-is reproduces a fixed point of the chain. `rust-checks`, `oci` run in parallel; `apple` follows `oci` so it sees the same trust root the linux cli was built against. A separate `cla.yml` workflow verifies that every PR author/committer email is covered by the current `CLA-INDIVIDUAL.md` or `CLA-CORPORATE.md` hash recorded in `CLA-SIGNERS.txt`. A separate `tinfoil-build.yml` workflow runs on `v*` tags and has two responsibilities: (1) generate `tinfoil-deployment.json` from `artifact-manifest.json`, attest it to Sigstore via `actions/attest`, and create the GitHub release — this is the artifact Tinfoil's verifier chain consumes; (2) sign `artifact-manifest.json` with `cosign sign-blob` (Fulcio keyless via the workflow's OIDC identity) and upload the manifest + its Sigstore bundle as release assets for the client's self-update verifier. The filename `tinfoil-build.yml` is mandated by Tinfoil's closed-source deployment system — do not rename. The release is intentionally not marked `latest` here; the release engineer's tooling does that once their human attestation is signed and uploaded.

**Image tagging:** `main` (rolling, updated on every merge), `v*` (immutable release tags), `sha-<short>` (per-commit). No `:latest`. Images published to `ghcr.io/<owner>/eidola-server`, `ghcr.io/<owner>/eidola-cli`, and `ghcr.io/<owner>/eidola-postgres`.

**Key design decisions:**

- Axum-based HTTP server with typed routing, extractors, and `utoipa-axum` OpenAPI integration
- Plain HTTP internally; TLS terminated by Tinfoil Container shim with attestation-bearing certificates (attestation hash + HPKE key encoded in SANs, issued by public CA)
- Tinfoil attestation verification via `tinfoil-verifier` crate — verifies SEV-SNP hardware attestation per-connection, caching verified fingerprints for fast reconnections; handles load-balanced deployments
- Deterministic enclave measurement via `measure-enclave` crate — pre-computes SEV-SNP and TDX measurements from source, written to `releases/trust/server-enclave.json` (the cli build input) and recorded in `artifact-manifest.json` (the signed deployment record)
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
- `TINFOIL_REPO` (optional) - Source repository the upstream enclave is attested against via the Tinfoil ATC `POST /attestation` endpoint (default: `tinfoilsh/confidential-model-router`); must match the GitHub repo whose signed measurements correspond to the running enclave
- `TINFOIL_PRICING_OVERRIDES` (optional) - JSON object overriding per-model pricing; e.g. `{"kimi-k2-6":{"input":2.0,"output":6.0}}`. Token-based models accept `input`/`output` ($/M tokens); per-request models accept `request` ($/request). See `backend.rs` `MODEL_CATALOG` for defaults
- `PRICING_MARKUP` (optional) - Pricing markup factor applied to all model prices (default: `1.5`)
- `OTEL_EXPORTER_OTLP_ENDPOINT` (optional) - OTLP endpoint; enables OpenTelemetry export of traces, metrics, and logs when set (e.g. `https://otlp-gateway-prod-us-central-0.grafana.net/otlp`)
- `OTEL_EXPORTER_OTLP_HEADERS` (optional) - OTLP auth headers (e.g. `Authorization=Basic <base64(instanceID:apiKey)>`)
- `OTEL_SERVICE_NAME` (optional) - Override service name in telemetry (default: `eidola-server`)

**Observability:**

The server uses OpenTelemetry to ship traces, metrics, and logs directly to Grafana Cloud (or any OTLP endpoint) via HTTP/protobuf. Enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set; otherwise only stdout logging. Telemetry respects the privacy boundary between the "linked" account layer and the "unlinked" anonymous service layer: chat completion spans/metrics contain only model name, token counts, status, and latency — never account IDs, credential data, or message content. Account layer spans may include account_id. The middleware (`middleware.rs`) classifies routes and creates per-request spans; metric instruments are defined in `telemetry.rs`.

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

The binary uses `sev` (with `crypto_nossl` feature ��� pure Rust, no OpenSSL) for SEV-SNP launch digest computation and `tdx-measure` for TDX RTMR1/RTMR2 runtime measurements. Both work natively on macOS. Output JSON matches the Tinfoil deployment manifest predicate (`snp-tdx-multiplatform/v1`): `{snp_measurement, tdx_measurement: {rtmr1, rtmr2}, cmdline}`.

`tinfoil-config.yml` is the Tinfoil Container configuration. It references container images by digest (from `artifact-manifest.json`), declares `_HASH` env vars for measured secrets (Argon2id hashes generated via `cargo run -p hash-secret`), and specifies CVM resources (cpus, memory). The SHA-256 of this file is embedded in the kernel command line and bound into the enclave measurement, so any change to the config produces a different measurement.

The measurement flow: `source → deterministic OCI build → server digest → tinfoil-config.yml (with digest) → cmdline (with config hash) → measurement → releases/trust/server-enclave.json → cli build embeds it as its trust root → cli OCI/macOS narHash → artifact-manifest.json`. The `server-enclave.json` step exists to break the otherwise-circular self-reference that would happen if the cli build COPYed the manifest containing its own digest; isolating the enclave fields lets the cli build see a stable input even as the manifest is regenerated. All values are committed and verified by CI. CVM artifacts are cached locally at `~/.cache/eidola/cvm/`. Pass `--verify-attestations` to `artifact-manifest.sh` (used by CI) to additionally verify CVM manifest provenance via Sigstore (`gh attestation verify --deny-self-hosted-runners`); this fails hard if verification fails.

**Tinfoil attestation verification (`crates/tinfoil-verifier/`):**

The `tinfoil-verifier` crate exposes `attesting_client()`, which returns a `reqwest::Client` that re-verifies enclave attestation on every new TCP+TLS handshake. There is no startup bootstrap and no verified-fingerprint cache: `attesting_client()` performs no network I/O, the *first* real request through the returned client is also the first attestation, and every subsequent new TLS handshake re-attests independently. Policy changes (TCB floor, allowed measurements) take effect on the next handshake without a process restart. Callers that want fail-fast-at-startup semantics issue a single throwaway request through the client immediately after construction and treat its outcome as the readiness check (the eidola server does this by hitting `{base}/models`).

**Per-handshake flow.** The client's connector layer (`AttestingConnectorLayer`) wraps reqwest's inner connector. On every new TCP+TLS handshake, after the TLS layer completes, the connector generates a fresh random 32-byte nonce and issues an inline HTTP/1.1 `GET /.well-known/tinfoil-attestation?nonce=<hex>` over the **same** connection (using `httparse` for response parsing and `hyper-util::TokioIo` to bridge `hyper::rt::Read/Write` to `tokio::io::AsyncRead/Write`). The enclave responds with a *freshly collected* hardware report whose `REPORT_DATA` is `SHA-256(tls_key_fp ‖ hpke_key ‖ nonce ‖ gpu_hash ‖ nvswitch_hash)` (the inference router is CPU-only, so the GPU/NVSwitch hashes are empty), plus the PEM TLS leaf cert and an ECDSA signature over the document. The fresh document does **not** carry the VCEK, so the connector consults Tinfoil's ATC service (`POST /attestation` with `{enclaveUrl, repo}`) over a side channel to backfill it (the shim mock self-carries the VCEK, so tests need no ATC). It then verifies, in `bundle.rs` + `attesting_client.rs`: the echoed nonce equals the one sent (freshness); `report_data.tls_key_fp == sha256(SPKI(peer_cert))` and the document's embedded cert matches the peer cert; the document's ECDSA signature against that cert (P-384 in production, P-256 for the mock, both SHA-256-prehash — signature reconstructed by blanking the `signature` value in the raw served bytes so unknown fields survive); the AMD VCEK chain (ARK → ASK → VCEK, RSA-PSS SHA-384); SEV-SNP / TDX report signature; TCB policy (bl≥0x07, snp≥0x0e, ucode≥0x48); measurement against `ALLOWED_MEASUREMENTS` in `measurements.rs`; and finally that the hardware report's `REPORT_DATA` equals the recomputed `SHA-256(tls_key_fp ‖ hpke_key ‖ nonce ‖ …)` — which authenticates every claimed `report_data` field against the AMD/Intel signature. Only if all of that passes is the connection yielded to hyper for the real request. ALPN is pinned to `http/1.1` so the inline attestation request and the subsequent application request share one HTTP lifecycle. Subsequent HTTP requests on a pooled keepalive connection do not re-trigger the connector and inherit the binding to the TLS key (and the nonce-bound report) that was attested when the connection was first established. The same-connection guarantee makes this safe behind load balancers: whatever backend the LB routes you to is the backend you attest.

The enclave's own fresh `/.well-known/tinfoil-attestation?nonce=<hex>` document is the source of truth. ATC is the single fallback target for the chain itself (today: the VCEK), and the legacy static `?v=3` / v2 attestation documents are no longer used. The verifier *does* fetch AMD KDS CRLs in production mode for revocation checks, but this is gated on `trusted_ark_der.is_none()` — test deployments that supply a custom mock ARK skip CRL fetching entirely (AMD KDS has no revocation entries for mock chips and the CRL signature would fail to verify against the mock ARK anyway). `trusted_ark_der` / `trusted_ask_der` only feed the SEV-SNP attestation chain verifier; they are **not** added to any TLS root store.

TLS trust roots are supplied by the caller via `AttestingClientConfig::tls_roots` and used for **all** outbound HTTPS the verifier performs (attested inference endpoint, ATC fallback, AMD KDS CRL fetches). The verifier crate intentionally does not depend on `webpki-roots` or `rustls-native-certs` so each consumer can pick the source that fits its environment without dragging the wrong dep into the others: the server runs `FROM scratch` inside an enclave with no system trust store and supplies `webpki-roots`; the CLI and macOS app supply `rustls-native-certs` so developers can install local dev CAs (e.g. the tinfoil shim mock's `tls-ca.pem`) in their OS keychain. The verifier crate's pure-Rust deps (`sev`, `x509-cert`, `der`, `tower`, `hyper`, `hyper-util`, `httparse`) — no OpenSSL.

The per-handshake nonce guarantees *freshness* — the enclave folds our random nonce into a freshly collected `REPORT_DATA`, so a stale or captured document can't be replayed against a different nonce, and we know a live, genuine CC machine currently holds the cert key. It does **not** by itself defeat exfiltration of the TLS key: the report binds the long-term TLS *key* (cert SPKI), not the live TLS *session*, so an attacker holding the stolen key could actively MITM (deriving the session keys despite TLS 1.3 forward secrecy) and relay a fresh nonce-bound report from the enclave's public endpoint, passing every client check. Closing that requires channel binding (a TLS-session/exporter value in `report_data`); today it still rests on the key staying sealed in the enclave. Separately, the fresh document is not yet *fully* self-contained — it omits the VCEK — so the ATC fallback remains in place; once Tinfoil folds the VCEK into the document, that path will go cold and can be removed entirely.

**Compose files:**

- `compose.yaml` — local development. Two supported workflows share one file:
  - **Full container stack** (`just dev`, → `scripts/dev.sh --container`) — postgres + server + shim + stripe-cli all in containers, detached. Server image is rebuilt each invocation.
  - **Host mode** (`just services`, → `scripts/dev.sh --host`) — postgres + shim + stripe-cli in containers, with `SHIM_UPSTREAM_URL=http://host.docker.internal:8080` so the shim forwards to a cargo-built server running on the host. Writes `.env.local` (`STRIPE_WEBHOOK_SECRET` + `BIND_ADDR=0.0.0.0:8080`) for the host server to source.
  - Both modes share `scripts/dev.sh`. Both build only the images they need, idempotently apply `schema.sql` to postgres, capture the Stripe webhook secret if `STRIPE_API_KEY` is set (otherwise skip stripe-cli), and start everything detached. `just down` tears down both modes.
  - Profiles: `server` gates the eidola-server container; `stripe` gates stripe-cli. Postgres and shim have no profile (always available). The shim has `extra_hosts: host.docker.internal:host-gateway` so the host-gateway alias works on Linux too, and intentionally has no `depends_on: server` so it starts cleanly in host mode. `postgres`, `server`, and `shim` declare `platform: linux/amd64` so compose doesn't warn on arm64 hosts about the (intentional) amd64 base layers.

## App Core Architecture

The GUI app and CLI share a common Rust core (`crates/eidola-app-core/`) consumed as a normal library — no FFI layer. All business logic — config management, local database, HTTP client construction, account operations, wallet/credential management, and chat inference — lives in the core crate. Consumers construct an `AppCore` and call its methods directly.

**Core crate modules:**

- `lib.rs` — `AppCore` struct, all high-level operations (account create/show/allocate, chat, wallet), DTO record types (`ConfigState`, `ChatResult`, `PriceInfo`, etc.), internal helpers (ACT token serialization, attestation flushing, HTTP response handling)
- `config.rs` — `Config` struct (TOML serde) with `*_override` fields and resolver methods that fall back to the embedded trust-root pin, load/save with explicit paths, measurement parsing, certificate parsing
- `trust_root.rs` — re-exports the build-time-generated `trust_root.gen.rs` constants (server URL, server enclave measurement, attestant fingerprints, CI identity, schema versions, embedded JSON for attestation templates + Sigstore trusted root). Source files live under `releases/`; see `docs/trust-root.md` for what's pinned and `releases/README.md` for how it rotates.
- `db.rs` — Turso (libSQL) database layer with 3-layer schema (wallet, transport, semantic), migrations, all CRUD operations
- `error.rs` — `AppError` enum, request error classification (attestation vs network vs server)

**CLI usage (`crates/eidola-cli/`):** Depends on `eidola-app-core` as a regular Rust crate. Calls `AppCore::new(config_dir, data_dir)` and invokes methods directly.

**GUI usage (`crates/eidola-gui/`):** Native Rust gpui app. Depends on `eidola-app-core` as a regular crate; `core.rs` wraps `AppCore` in an `Entity<Core>` that bridges tokio (the core's runtime) to gpui's smol-based executor via `oneshot` channels, and holds cached snapshots that views read reactively. **See `crates/eidola-gui/AGENTS.md` for the full architecture** — window model, Circadian theme + Newsreader bundling, transparent titlebar / gradient overlay, macOS menu/keybinding setup and ordering invariants, the per-view `CloseWindow` / `Settings` singleton patterns, the `.app` bundling requirement, and the two-tier test model (behavior tests as the regression gate; visual snapshots as a local debug aid).

**Crate layout:** Pure Rust crates in `crates/` implement capability logic — `eidola-app-core`, `eidola-server`, `tinfoil-verifier`, `gpui-markdown-editor` — plus operational utilities such as `generate-openapi`, `tinfoil-shim-mock`, `hash-secret`, and `measure-enclave`. `crates/devtools/` is an anchor crate (package name `eidola-devtools`) whose sole purpose is to pull upstream Rust-based dev tools (currently `rumdl` and `just`) into the workspace dep graph for lockfile pinning; `.envrc` builds them and direnv puts them on `PATH`.

## Local Database & Migrations

Both the CLI and GUI use an embedded [Turso](https://crates.io/crates/turso) (pure-Rust libSQL) database at `~/Library/Application Support/eidola/eidola.db` for local app data (wallet credentials, conversation history, attestation records, etc.). The database layer lives in `crates/eidola-app-core/src/db.rs`.

**Schema management:**

- `crates/eidola-app-core/schema/schema.sql` is the canonical schema — always reflects the current desired state
- Fresh installs apply `schema.sql` directly via `execute_batch` and set `PRAGMA user_version` to `LATEST_VERSION`
- Existing databases run incremental migrations in `db.rs` (gated by `user_version`)

**Adding a migration:**

1. Update `schema.sql` to the new desired state
2. Add a `MIGRATION_N` constant in `db.rs` with the ALTER/CREATE statements
3. Add an `if current_version < N` block in `migrate()` that runs the migration and sets `user_version`
4. Bump `LATEST_VERSION`
5. Run `cargo test -p eidola-app-core` — the `migrations_match_schema` test structurally compares a fresh-from-schema database against a fully-migrated database (via `PRAGMA table_info`, `PRAGMA index_info`, and view SQL)

**Limitations:** The `turso` crate does not support `ALTER TABLE ALTER COLUMN` (a libSQL C extension). To add `NOT NULL` columns, use `ADD COLUMN ... DEFAULT <value>` — the default persists and must also be declared in `schema.sql` so both paths match.

## Build Commands

**Prerequisites:** `rustup`, `direnv`, `docker`

The `justfile` is the primary development interface. Run `just` to see all available recipes. `just` itself is not a separate install: it is one of the Rust-based dev tools pinned by the `crates/devtools/` anchor crate, built by `.envrc`, and put on `PATH` by direnv — so `rustup` + `direnv` is enough to get it (see Conventions for how the anchor works).

**Key recipes:**

- `just build {server,cli,gui}` — local-toolchain builds for fast iteration. The `gui` target on macOS additionally runs `scripts/package-gui-app.sh` to assemble `crates/eidola-gui/build/Eidola.app` (the .app wrapper is required for AppKit to treat the binary as a real app rather than a command-line tool — see `crates/eidola-gui/AGENTS.md` for why).
- `just run {server,cli,gui}` — build and run. For `gui`, opens the assembled `.app` via `open`. Accepts trailing args (e.g. `just run cli chat "hello"`).
- `just test` — runs `cargo test`.
- `just check` — clippy, rustfmt, and `rumdl check` over committed markdown (see `.rumdl.toml`). Run `just lint-md-fix` to apply auto-fixable formatting.
- `just dev` / `just services` / `just down` — container-based development workflows (see Compose files above).

## Conventions

- Pure Rust dependencies preferred (for cross-compilation)
- Keep Rust workspace packages under `crates/`; do not add a separate top-level `tools/` tree
- `just` is the task runner — wrap scripts and common commands as recipes
- Server and CLI OCI images are built with StageX (reproducible, `FROM scratch`, runs as non-root)
- Nix is used for CI quality gates and the reproducible macOS universal-binary builds (CLI binary + GUI `.app` bundle), not daily Rust development
- `rustup` + `rust-toolchain.toml` manages the Rust toolchain for development
- OpenAI API format as the canonical interface
- Server API is documented via utoipa `#[utoipa::path]` annotations on handler functions and `ToSchema` derives on request/response types. `OpenApiRouter` (in `lib.rs::build_router()`) collects paths and recursively discovers schemas automatically — only SSE streaming types that aren't referenced from path annotations are listed manually in `api_doc.rs`. When adding or changing server endpoints, add the annotation on the handler and register it in `build_router()` via `routes!()`, then run `just update-openapi` to regenerate the committed `openapi.json`
- Rust-based dev tools are version-pinned in the workspace `Cargo.lock` through the `crates/devtools/` anchor crate (package name `eidola-devtools`, lib-only, no code — just `=X.Y.Z` deps so each upstream tool is in our dep graph). `.envrc` runs `cargo build --quiet -p rumdl -p just` and direnv puts `target/debug/` on `PATH`, so a `rustup` + `direnv` checkout has the pinned `just`, `rumdl`, etc. with no separate install. `cargo build -p <tool>` resolves unambiguously to the upstream crate because the anchor's package name differs from every tool it anchors; only crates with a `lib` target can be anchored (a bin-only crate cannot be a `[dependencies]` entry — `just` and `rumdl` both qualify). Adding a tool is one `=X.Y.Z` dep in `crates/devtools/Cargo.toml` plus a `-p <tool>` in `.envrc`; bumping one is a single edit there. Markdown specifically is linted via [rumdl](https://rumdl.dev/): `just check` runs `rumdl check .` (rumdl is already on `PATH` via direnv). CI deliberately does not install direnv (no apt in the build chain); it emulates `.envrc` by building only rumdl and prepending `target/debug` to `PATH`, so the `rumdl check .` invocation matches the local one without compiling the rest of the pinned toolset. Configure rules in `.rumdl.toml`
- `artifact-manifest.json` (`schema_version: 1` — integer; see `docs/trust-root.md` for why our schema versions are integers, not semver) records expected OCI digests, the macOS universal CLI binary and GUI `.app` bundle Nix `narHash`es, and enclave measurements (SEV-SNP + TDX + cmdline) with type/platform metadata; the enclave block shape matches the Tinfoil `snp-tdx-multiplatform/v1` predicate so `tinfoil-build.yml` can project it directly into `tinfoil-deployment.json`. CI verifies the full file by merging digests captured from the real OCI and macOS build jobs and recomputing enclave measurements from `tinfoil-config.yml`
- `releases/trust/server-enclave.json` (`schema_version: 1`) holds the same enclave block in isolation — `snp_measurement`, `tdx_measurement: {rtmr1, rtmr2}`, `cmdline`. It is the only build-time input that ties the cli to the server, intentionally separated from `artifact-manifest.json` so the cli build context doesn't drag its own digest into its own inputs. `verify-full` cross-checks both files against the freshly-recomputed enclave so neither can drift unobserved
- Use `just update-manifest` to regenerate both on macOS with the pinned amd64 BuildKit builder plus the local Nix CLI build. The script runs a two-phase build (server + postgres → stamp tinfoil-config.yml → recompute enclave → write `server-enclave.json` → cli OCI + cli macOS → write `artifact-manifest.json`), so a single invocation reaches a fixed point
- Contributor agreement state lives in `CLA-INDIVIDUAL.md`, `CLA-CORPORATE.md`, and `CLA-SIGNERS.txt`; the signer ledger plus Git history is the source of truth, and changing either CLA text requires new signer entries because the SHA-256 hash changes
- Before committing, ensure `README.md` and the relevant `AGENTS.md` are updated to reflect any changes. Workspace-wide context (server, app-core, build commands, conventions) goes in the top-level `AGENTS.md`; gpui-app-specific context goes in `crates/eidola-gui/AGENTS.md` (loaded automatically when working in that subtree). Sub-app docs should not duplicate workspace-wide context — link back to it instead.
- Omit any tool-specific "co-authored by" lines from commit messages
