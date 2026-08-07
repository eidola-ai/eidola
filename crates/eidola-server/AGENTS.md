# eidola-server — Agent Development Guide

An OpenAI-compatible proxy that translates requests to upstream AI providers, with a billing system using anonymous credentials for privacy-preserving usage tracking.

- **Current upstream:** Tinfoil (inference.tinfoil.sh) — OpenAI-compatible; all models run in confidential enclaves (AMD SEV-SNP / Intel TDX / NVIDIA CC).
- **Database:** PostgreSQL 17+ (`schema/schema.sql`).
- **Deployment:** Tinfoil Containers — all services run inside confidential enclaves. The Tinfoil shim terminates TLS with attestation-bearing certificates; the server runs plain HTTP behind it.
- **API endpoints:** defined in `openapi.json`, generated from utoipa annotations — see Conventions in the top-level AGENTS.md (`just update-openapi`).

## Key design decisions

- Axum HTTP server with typed routing, extractors, and `utoipa-axum` OpenAPI integration.
- Plain HTTP internally; TLS terminated by the Tinfoil shim (attestation hash + HPKE key encoded in SANs, issued by a public CA).
- Tinfoil attestation verified per-connection via the `tinfoil-verifier` crate (see that crate's AGENTS.md).
- Deterministic enclave measurement via `measure-enclave` (see that crate's AGENTS.md) → `releases/trust/server-enclave.json` (the cli build input) + `artifact-manifest.json` (the signed deployment record).
- Statically linked musl binaries; StageX-based OCI images (reproducible, `FROM scratch`, non-root).
- Request-based — no sessions/caching in the proxy layer.
- Account auth (Basic + Argon2id) via `BasicAuth` extractor; chat completions auth via `TokenAuth`.
- **Tool calling is wire-level pass-through** (`types.rs`): the request accepts `tools` / `tool_choice`, an assistant `tool_calls` with nullable `content`, and `role: "tool"` + `tool_call_id`; the response and SSE chunk relay `message.tool_calls` / `delta.tool_calls` and `finish_reason: "tool_calls"`. The server **never executes a tool** — the loop lives in app-core. Each new field is a raw `serde_json::Value`, deliberately: the proxy re-serializes everything it parses, so a narrower struct would silently drop provider extension fields and the streaming-only `index` framing key the client's accumulator needs (the same defect the `reasoning_content` / `reasoning` fields were added to fix). `content` is `Option` with **no** `skip_serializing_if`, so a deliberate `"content": null` survives to the upstream chat template. The strict `deny_unknown_fields` on the request stays: the server remains the authority on what it accepts; opacity is scoped to the *inside* of tool payloads. No "supports tools" capability bit — an upstream 4xx pass-through is the honest v1 error.
- Stripe integration via a thin `reqwest` wrapper (no `async-stripe`).
- **Pricing contract:** the server recomputes `eidola-common::prompt_charge` — the identical function of the identical `messages`/`tools` arrays the client computes — as its pre-flight minimum, and clamps charged prompt tokens to it, so the client's hold covers the charge by construction. Tool bytes count (every `tool_calls` entry's and tool schema's compact JSON serialization) at the same safe cost factor.

## Environment variables

- `TINFOIL_API_KEY` (required) — Tinfoil inference API key.
- `DATABASE_URL` (required) — PostgreSQL connection string.
- `DATABASE_PASSWORD` (optional) — in production with an external database, inject as a Tinfoil secret instead of embedding in `DATABASE_URL`.
- `DATABASE_SSL_CERT` (optional) — PEM root CA for PostgreSQL TLS verification; public material, normal env var.
- `CREDENTIAL_MASTER_KEY` (required) — hex-encoded 32-byte AES-256 key encrypting issuer private keys at rest in Postgres. Production: a Tinfoil secret; local dev: the all-zeros key from `.env.example`. Must remain stable across upgrades so encrypted issuer keys stay accessible.
- `BIND_ADDR` (default `127.0.0.1:8443`) — HTTP bind address; Containerfile overrides to `0.0.0.0:8080`.
- `STRIPE_API_KEY` / `STRIPE_WEBHOOK_SECRET` (optional) — billing endpoints / webhook return 503 without them.
- `TINFOIL_BASE_URL` (optional) — override the default `https://inference.tinfoil.sh/v1`.
- `TINFOIL_REPO` (optional) — source repo the upstream enclave is attested against via the Tinfoil ATC `POST /attestation` endpoint (default `tinfoilsh/confidential-model-router`); also the repo whose latest release the runtime measurement resolver (`src/upstream_trust`) verifies.
- `TINFOIL_MEASUREMENT_REFRESH_SECS` (optional, default `600`) — how often `src/upstream_trust` re-checks Tinfoil's latest release. Bounds the release→deploy race window; 10m stays well within GitHub's unauthenticated rate limit.
- `TINFOIL_PRICING_OVERRIDES` (optional) — JSON per-model pricing overrides, e.g. `{"kimi-k2-6":{"input":2.0,"output":6.0}}`. Token-based models take `input`/`output` ($/M tokens); per-request models take `request`. Defaults in `backend.rs` `MODEL_CATALOG`.
- `PRICING_MARKUP` (optional, default `1.5`) — markup factor on all model prices. The server refuses to start below 1.5 — the pricing contract's safe cost factor (`eidola-common`'s `SAFE_COST_FACTOR_NUM/DEN = 3/2`): the contract charges prompt bytes at 1/1.5 of worst-case token count, so a lower markup would sell tokens below cost (`validate_pricing_markup` in `backend.rs`).
- `TERMS_FEED_BASE_URL` (optional) — base URL of the published website (production `https://www.eidola.ai`) the terms-feed poller (`src/terms_feed.rs`) polls for current legal-document versions. See Terms gate below.
- `TERMS_REFRESH_SECS` (optional, default `600`) — terms-feed poll interval.
- `TERMS_OF_SERVICE_SHA256` / `PRIVACY_POLICY_SHA256` (+ optional `_VERSION`, default 1, and `_URL`) — static dev/test pins seeded once at startup through the same monotonic upsert. Neither feed nor pins set = gate disabled.
- `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` / `OTEL_SERVICE_NAME` (optional) — OpenTelemetry export (see Observability).

## Terms gate (`src/terms_feed.rs`)

Each legal document's exact source bytes are published by the site builder at `/terms/source.md` and `/privacy/source.md` (rendered pages carry `eidola:version` / `eidola:source-sha256` meta tags and a visible version line). The server computes the SHA-256 itself, parses the front-matter `version`, and records observed versions in the shared `required_document` table — **append-only**, one row per (document, version) with `first_required_at`. The gate enforces the highest recorded version per document, so monotonicity is structural (a stale CDN edge or outage can only pause advancement, never regress it) and the table is a permanent audit trail. A same-version observation with different bytes is a versioning-contract violation and refused (warn from the poller; fatal for an explicit env seed). The table is shared, so one instance observing a new version converges every instance immediately. This decouples terms updates from the measured, per-release-immutable server config — legal documents outlive any release.

**Gate semantics:** checkout and credential issuance require the account's max accepted version per document ≥ the required version (HTTP 428 `terms_acceptance_required`; clients fetch `GET /v1/terms`, record acceptance via `POST /v1/account/terms`, stored append-only in `account_acceptance` with the version stamped). `AppCore::account_create` records acceptance after the UI captures consent (GUI onboarding checkbox; CLI `account create --accept-terms`); `AppError::TermsAcceptanceRequired` routes re-acceptance.

CI enforces the ordering contract: any PR changing a legal document's bytes must increment its front-matter version by exactly 1 (`scripts/check-legal-doc-versions.sh`, wired into `rust-checks.yml`).

## Observability

OpenTelemetry ships traces, metrics, and logs directly to Grafana Cloud (or any OTLP endpoint) via HTTP/protobuf, enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set; otherwise stdout logging only. Telemetry respects the privacy boundary between the "linked" account layer and the "unlinked" anonymous service layer. `middleware.rs` classifies routes and creates one span per request; instruments live in `telemetry.rs`. Per-signal doctrine (and the reasoning) is in [docs/server.md](../../docs/server.md#telemetry-scope-and-boundary), enforcing `privacy-guarantees.md` §3.2–3.3:

- **Spans** carry route, method, status, latency, and layer — nothing content-derived. Account-layer spans may include `account_id`; chat spans carry no identifier at all.
- **Metrics** are the only home for content-derived quantities (token counts), and only aggregated across requests. Never use a caller-supplied string as a label — `lookup_model` resolves the requested id against the known model list first, which is what keeps label cardinality bounded.
- **`Display` on `ServerError` is the log-safe rendering, not the full one** — `to_error_response` is the full detail and goes only to the client. `PaymentRequired` and `Backend` deliberately omit their `message` because those carry, respectively, a function of the prompt's chargeable bytes and an upstream-authored string. Unit tests in `error.rs` pin this. A new error message that interpolates anything request-derived must join the redacted set.
- Log call sites take `{e}` on a `ServerError` freely — the redaction lives in the type, not in the call site. Do not hand-format an error's fields into a log instead.

## Tinfoil Containers / TEE integration

The server runs plain HTTP inside a Tinfoil Container; the shim terminates TLS externally with attestation-bearing certificates and serves `/.well-known/tinfoil-attestation` for client-side verification.

- `CREDENTIAL_MASTER_KEY` is injected as a Tinfoil secret (encrypted, enclave-only) in production.
- With an external PostgreSQL (until Tinfoil supports persistent disks): connection metadata in `DATABASE_URL`, `DATABASE_PASSWORD` as a Tinfoil secret, `DATABASE_SSL_CERT` if the server cert doesn't chain to a WebPKI root.
- The container has `/dev/sev-guest` (via the undocumented `devices` field in `tinfoil-config.yml`) for SEV-SNP attestation reports; the pre-generated attestation document and TLS key material are at `/tinfoil/`.

`tinfoil-config.yml` (workspace root) is the Tinfoil Container configuration: image digests from `artifact-manifest.json`, `_HASH` env vars for measured secrets (Argon2id hashes via `cargo run -p hash-secret`), CVM resources. Its SHA-256 is embedded in the kernel command line and bound into the enclave measurement — any change produces a different measurement.

## Upstream measurement resolution (`src/upstream_trust/`)

`inference.tinfoil.sh` is a *router* enclave reverse-proxying to per-model GPU enclaves; the router trusts those downstream enclaves via "latest Sigstore-signed release of the repo". Statically pinning the router measurement bought little rigor over Tinfoil's own model while causing a fail-closed outage on every Tinfoil router release (their sign→deploy lag is ~zero; ours was hours). So the server resolves the allowed router measurement **at runtime** to match Tinfoil's actual trust level, while still running our superior per-handshake, nonce-fresh attestation on every request. **This subsystem is temporary — deleted when we self-host inference**, at which point `main.rs` hands `attesting_client` a statically pinned measurement set.

- It never touches `tinfoil-verifier` (whose connector bakes the allowed set at construction): to change the set we rebuild the `reqwest::Client` and hot-swap via `arc_swap::ArcSwap` (backend reads lock-free per request; `main.rs` supplies an `AttestingClientFactory` closure so this module stays telemetry-free).
- `sigstore.rs` verifies the GitHub artifact attestation (a Sigstore **`dsse`/in-toto** bundle — distinct from the `hashedrekord` blob bundle app-core's `ci_sigstore` verifies) end-to-end: Fulcio chain + identity (GitHub Actions OIDC issuer + repo **and exact tag** — tighter than Tinfoil's own clients), DSSE PAE signature, Rekor SET + RFC 6962 inclusion proof, `sha256(tinfoil-deployment.json) == subject digest`; the measurement is read from the signed `snp-tdx-multiplatform/v1` predicate. Crypto primitives ported from app-core's `ci_sigstore` (same pinned `releases/trust/sigstore-trusted-root.json`, the sole `releases/` input the server `build.rs` embeds); the SCT + Rekor-checkpoint gaps in `docs/gaps.md` apply identically.
- **No static fallback:** bootstrap resolves + verifies the latest release synchronously as the readiness gate; failure means the server refuses to start. Bootstrap's initial allowed set is a rolling window of 2 — latest plus the previous **published** release — so a cold start mid rolling-deploy still attests the draining previous router (the previous entry is best-effort; a missing/unverifiable one warns and boots latest-only).
- The periodic refresh keeps a rolling window of the last two measurements and never clears or widens trust on error: a new measurement is adopted only after its replacement client builds successfully; a failure keeps the current client (requests fail closed until the next successful tick).
- End-to-end regression: `tests/upstream_trust_verify.rs` verifies a real captured `v0.0.115` release bundle and asserts the tag/repo/digest/tamper rejections.

## Compose files (local development)

`compose.yaml` — two supported workflows share one file:

- **Full container stack** (`just dev` → `scripts/dev.sh --container`): postgres + server + shim + stripe-cli in containers, detached. Server image rebuilt each invocation.
- **Host mode** (`just services` → `scripts/dev.sh --host`): postgres + shim + stripe-cli in containers, with `SHIM_UPSTREAM_URL=http://host.docker.internal:8080` so the shim forwards to a cargo-built server on the host. Writes `.env.local` (`STRIPE_WEBHOOK_SECRET` + `BIND_ADDR=0.0.0.0:8080`) for the host server to source.

The shim service forwards `DEV_MEASUREMENT` as a bare pass-through (`- DEV_MEASUREMENT`, no `=`), so it reaches the container only when set in the environment running compose or in `.env` (which compose reads from the project directory automatically) — an unset var leaves the shim on its all-zeros default rather than an empty measurement, which would panic it. `scripts/local-client.sh` resolves the same variable the same way, environment before `.env`, so one value governs both what the shim advertises and what the client trusts on every documented path. Its `.env` reader covers compose's grammar except `${VAR}` interpolation (reimplementing that is reimplementing compose): such a value warns and then fails the script's own shape check, pointing at exporting the variable instead. The script's other settings (`EIDOLA_DEV_CERT_DIR` / `EIDOLA_DEV_BASE_URL`) are client-side only by design — they say where a stack already is, and have no compose counterpart to follow.

Both build only the images they need, idempotently apply `schema.sql`, capture the Stripe webhook secret if `STRIPE_API_KEY` is set (else skip stripe-cli), and start detached. `just down` tears down both. Profiles: `server` gates the server container, `stripe` gates stripe-cli; postgres and shim have no profile. The shim has `extra_hosts: host.docker.internal:host-gateway` (Linux host-gateway) and intentionally no `depends_on: server` (host mode); `postgres`/`server`/`shim` declare `platform: linux/amd64` so compose doesn't warn on arm64 hosts.
