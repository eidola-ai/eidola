# AGENTS.md

Guidance for AI coding agents working in this repository.

Eidola is a privacy-preserving AI chat system: an OpenAI-compatible proxy server running in confidential enclaves, plus a native GUI app and CLI sharing one Rust core. This file is the workspace orientation; **deep, area-specific doctrine lives in scoped AGENTS.md files that load when you work in that subtree** — read the relevant one before changing anything non-trivial there:

| Area | Doc |
|---|---|
| Shared app core (chat path, participants, wallet, local DB, local models, memory, tools) | `crates/eidola-app-core/AGENTS.md` |
| Server (endpoints, env vars, billing, TEE integration, upstream trust) | `crates/eidola-server/AGENTS.md` |
| GUI (gpui app: windows, theme, state doctrine, a11y, testing) | `crates/eidola-gui/AGENTS.md` + `crates/eidola-gui/STATE.md` |
| Markdown editor widget | `crates/gpui-markdown-editor/AGENTS.md` |
| CLI (commands, systemd service doctrine) | `crates/eidola-cli/AGENTS.md` |
| Attestation verification (per-handshake flow, threat model) | `crates/tinfoil-verifier/AGENTS.md` |
| Enclave measurement (the measurement flow) | `crates/measure-enclave/AGENTS.md` |
| CI, branch flow, release artifacts, caching | `.github/AGENTS.md` |
| Website generator | `crates/eidola-www/AGENTS.md` |
| Design/threat-model background (published docs) | `docs/` |

## Architecture at a glance

**Server** (`crates/eidola-server`): Axum-based OpenAI-compatible proxy to Tinfoil (`inference.tinfoil.sh`); PostgreSQL 17+; anonymous-credential billing (Stripe via a thin `reqwest` wrapper); deployed as a StageX `FROM scratch` image inside Tinfoil Containers (AMD SEV-SNP), plain HTTP behind the attestation-bearing TLS shim. Tool calling is wire-level pass-through — the server never executes a tool. The client and server both compute the pricing contract from `eidola-common` (`prompt_charge`), so the client's hold covers the server's charge by construction.

**App core** (`crates/eidola-app-core`): the GUI and CLI share this crate as a normal library — no FFI. All business logic lives here: config, the embedded Turso (libSQL) database, attested HTTP clients, accounts/wallet, chat inference (a bounded agentic tool loop), participants/templates, agent memory, local llama.cpp models. `AppCore::new` is fallible (exclusive advisory `flock` on the local DB — one writer, loudly; second opener gets `AppError::DatabaseInUse`). A `Change` bus broadcasts domain invalidations after every durable commit. **Any change to the chat path must extend the chat harness** (`tests/chat_path.rs` + the `tests/bus.rs` exit-point table).

**Local database:** fresh-start, no migrations — `crates/eidola-app-core/schema/schema.sql` is the whole baseline. Schema changes edit `schema.sql` and bump `LATEST_VERSION` in `db.rs`; other versions are refused ("delete your dev database"), never migrated. turso defaults `foreign_keys` OFF; `db::connect()` enables it per connection.

**Trust chain:** deterministic OCI build → server digest → `tinfoil-config.yml` → enclave measurement (`measure-enclave`) → `releases/trust/server-enclave.json` → embedded in the cli/GUI as their trust root → `artifact-manifest.json` (the signed deployment record). Clients re-verify enclave attestation on every TLS handshake (`tinfoil-verifier`). The server resolves the upstream router measurement at runtime from Tinfoil's Sigstore-signed releases (`src/upstream_trust` — temporary until we self-host inference).

**Website** (`crates/eidola-www` + `www/`): in-workspace static-site generator; docs render directly from `docs/`; adding a doc requires a `www/docs-nav.toml` entry (the build fails otherwise).

## Branch flow

`next` is the integration trunk; `main` is verified and promotion-only. **Never commit to `main` directly** — feature/dependency PRs target `next`; a maintainer promotes via `/promote` on the `next` → `main` PR, which fast-forwards `main` to the validated head. Cheap gates (`rust-checks`, lint, CLA, …) run on `next` PRs; the expensive `Build & Verify Artifacts` chain runs only for `main` (push + the promotion PR) and its `Guard` job is the merge gate. Full doctrine: `.github/AGENTS.md`.

## Build commands

**Prerequisites:** `rustup`, `direnv`, `docker`. The `justfile` is the primary development interface — run `just` to see all recipes. `just` itself is one of the Rust dev tools pinned via the `crates/devtools/` anchor crate, built by `.envrc`, and put on `PATH` by direnv.

- `just build {server,cli,gui,www}` — local-toolchain builds. `gui` on macOS also assembles `crates/eidola-gui/build/Eidola.app` (required for AppKit to treat the binary as a real app).
- `just run {server,cli,gui,www}` — build and run; accepts trailing args (e.g. `just run cli chat "hello"`). `www` serves locally with drafts + rebuild-on-change.
- `just test` — `cargo test`.
- `just check` — clippy, rustfmt, and `rumdl check` over committed markdown (`.rumdl.toml`); `just lint-md-fix` applies auto-fixes.
- `just dev` / `just services` / `just down` — container-based dev workflows (full stack vs. host-mode services; see `crates/eidola-server/AGENTS.md` → Compose files).
- `just client-local` / `just client-reset` — point this machine's client (the GUI and CLI share one profile) at the local dev stack, and revert. Dev-only convenience over the sanctioned override surface: `scripts/local-client.sh` shells out to `eidola configure`, which writes base URL + ARK/ASK + trusted measurement as per-column overrides on the `eidola` backend row; `client-reset` clears each back to NULL, which *is* the compiled-in trust-root pin. Attestation is never weakened — the client runs the same per-handshake verification against the mock shim's chain. Quit any running Eidola first (single-writer local DB). Trusting the shim's mock TLS root in the OS trust store needs `sudo` and stays a human step: the script prints the command, never runs it. **Only the trust bundle moves** — the account (config.toml) and the wallet (local DB) stay with the profile, and neither is per-column overridable, so `client-local` pre-flights the profile and warns specifically when either is present (`eidola account create` refuses while an account exists; a production credential can be picked for a local turn and parked until `eidola wallet credentials recover` after `client-reset`).
- `just update-openapi` — regenerate the committed `openapi.json` after changing server endpoints.
- `just update-manifest` — regenerate `artifact-manifest.json` + `releases/trust/server-enclave.json`. **Release-time only; do not run as part of a feature change** (the desktop narHashes move with any sidecar/toolchain change).
- `just engine` — materialize the bundled llama.cpp sidecar via Nix for the dev `.app` (optional; `just build gui` does not depend on Nix).

**Dev profile:** dependencies build at `opt-level = 2` in dev (`[profile.dev.package."*"]`) while our crates stay unoptimized — 8-17× faster GUI frames for ~90 s one-time cost. The cost is debugging *inside* a dependency (locals read "optimized out"); backtraces are unaffected. Measurements in the comment above the profile in the root `Cargo.toml`.

## Crate layout

Pure Rust crates under `crates/` (do not add a top-level `tools/` tree): `eidola-server`, `eidola-app-core`, `eidola-cli`, `eidola-gui`, `eidola-www`, `gpui-markdown-editor`, `tinfoil-verifier`, `measure-enclave`, and `eidola-common` — shared contract logic that must be bit-identical across boundaries: the pricing contract both client and server compute (`prompt_charge`, `chargeable_prompt_tokens`, `PromptCharge`) and the `embed` marker-recognition rule shared with the markdown editor. Its dependency rule: admissible only if already in every consumer's graph and required for contract fidelity (today just `serde_json`). `eidola-attestation` is the shared attestation-template rendering both `release-tool` (signing side) and app-core's updater (verifier side) must agree on character-for-character. Operational utilities: `generate-openapi`, `tinfoil-shim-mock`, `hash-secret`, `release-tool`. `crates/devtools/` is a lib-only anchor crate pinning Rust dev tools (`rumdl`, `just`) in `Cargo.lock`; adding a tool is one `=X.Y.Z` dep there plus a `-p <tool>` in `.envrc`.

## Conventions

- Pure Rust dependencies preferred (cross-compilation); OpenAI API format is the canonical interface.
- Server endpoints are documented via utoipa `#[utoipa::path]` annotations + `ToSchema` derives, collected by `OpenApiRouter` in `lib.rs::build_router()` (SSE-only types listed manually in `api_doc.rs`). New/changed endpoints: annotate, register via `routes!()`, run `just update-openapi`.
- Nix is used for CI quality gates and reproducible desktop release builds (macOS universal cli + GUI `.app`, Linux GUI), not daily development. The bundled llama.cpp `llama-server` sidecar (`packages.llama-server` in `flake.nix`, pinned to release b9960 for `gemma4` support, statically linked, trimmed to one binary) ships in every desktop artifact; app-core resolves it exe-relative and never scans `$PATH`. The macOS sidecar is deliberately arm64-only (Intel Macs get the honest missing-engine state); Linux ships CPU-only for now.
- Artifact/trust files: `artifact-manifest.json` (`schema_version: 1` — integer, see `docs/trust-root.md`) records OCI digests, desktop narHashes, and enclave measurements; `releases/trust/server-enclave.json` isolates the enclave block as the cli's build input. Regeneration and verification doctrine: `.github/AGENTS.md` and `crates/measure-enclave/AGENTS.md`.
- CLA state lives in `CLA-INDIVIDUAL.md`, `CLA-CORPORATE.md`, `CLA-SIGNERS.txt`; changing either CLA text requires new signer entries (the hash changes).
- **Before committing, update the relevant AGENTS.md** to reflect any changes. Placement: workspace-wide context (architecture summaries, build commands, conventions) belongs here; area-specific doctrine belongs in the scoped file for that subtree (see the table at the top). Keep this file lean — it is loaded into every session; detailed design rationale, invariants, and test indexes go in the scoped files. Sub-docs should not duplicate workspace-wide context — link back instead.
- Omit tool-specific "co-authored by" lines from commit messages.
