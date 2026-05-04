# AGENTS.md

Guidance for AI coding agents working in this repository.

## Server Architecture

The server is an OpenAI-compatible proxy that translates requests to upstream AI providers. It includes a billing system with anonymous credentials for privacy-preserving usage tracking.

**Current upstream:** Tinfoil (inference.tinfoil.sh) — OpenAI-compatible, all models run in confidential enclaves (AMD SEV-SNP / Intel TDX / NVIDIA CC)

**Database:** PostgreSQL 17+ (see `crates/eidola-server/schema/schema.sql`)

**Deployment:** Tinfoil Containers — all services run inside confidential enclaves (AMD SEV-SNP). The Tinfoil shim handles TLS termination with attestation-bearing certificates; the server runs plain HTTP behind it.

**CI:** `ci.yml` contains four jobs — `rust-checks` (ubuntu-24.04: cargo fmt/clippy/test, OpenAPI freshness), `oci` (ubuntu-24.04: OCI image builds, OCI subset verification, GHCR publishing), `apple` (self-hosted Mac: Swift formatting/bindings freshness, Nix-based macOS app and CLI universal binary builds, Swift tests), and `artifact-manifest` (ubuntu-24.04: merges the OCI and macOS artifact digests, recomputes enclave measurements from `tinfoil-config.yml` + CVM artifacts, and verifies the full committed manifest). The `oci` and `apple` jobs gate on `rust-checks` to avoid wasting resources on failing PRs. A separate `cla.yml` workflow verifies that every PR author/committer email is covered by the current `CLA-INDIVIDUAL.md` or `CLA-CORPORATE.md` hash recorded in `CLA-SIGNERS.txt`. A separate `tinfoil-build.yml` workflow runs on `v*` tags to generate `tinfoil-deployment.json` from `artifact-manifest.json`, attest it to Sigstore via `actions/attest`, and create a GitHub release — this is the artifact Tinfoil's verifier chain consumes.

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
- `TINFOIL_REPO` (optional) - Source repository the upstream enclave is attested against via the Tinfoil ATC `POST /attestation` endpoint (default: `tinfoilsh/confidential-model-router`); must match the GitHub repo whose signed measurements correspond to the running enclave
- `TINFOIL_PRICING_OVERRIDES` (optional) - JSON object overriding per-model pricing; e.g. `{"kimi-k2-5":{"input":2.0,"output":6.0}}`. Token-based models accept `input`/`output` ($/M tokens); per-request models accept `request` ($/request). See `backend.rs` `MODEL_CATALOG` for defaults
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

The measurement flow: `source → deterministic OCI build → digest → tinfoil-config.yml (with digest) → cmdline (with config hash) → measurement`. All values are committed in `artifact-manifest.json` and verified by CI. CVM artifacts are cached locally at `~/.cache/eidola/cvm/`. Pass `--verify-attestations` to `artifact-manifest.sh` (used by CI) to additionally verify CVM manifest provenance via Sigstore (`gh attestation verify --deny-self-hosted-runners`); this fails hard if verification fails.

**Tinfoil attestation verification (`crates/tinfoil-verifier/`):**

The `tinfoil-verifier` crate exposes `attesting_client()`, which returns a `reqwest::Client` that re-verifies enclave attestation on every new TCP+TLS handshake. There is no startup bootstrap and no verified-fingerprint cache: `attesting_client()` performs no network I/O, the *first* real request through the returned client is also the first attestation, and every subsequent new TLS handshake re-attests independently. Policy changes (TCB floor, allowed measurements) take effect on the next handshake without a process restart. Callers that want fail-fast-at-startup semantics issue a single throwaway request through the client immediately after construction and treat its outcome as the readiness check (the eidola server does this by hitting `{base}/models`).

**Per-handshake flow.** The client's connector layer (`AttestingConnectorLayer`) wraps reqwest's inner connector. On every new TCP+TLS handshake, after the TLS layer completes, the connector issues an inline HTTP/1.1 `GET /.well-known/tinfoil-attestation?v=3` over the **same** connection (using `httparse` for response parsing and `hyper-util::TokioIo` to bridge `hyper::rt::Read/Write` to `tokio::io::AsyncRead/Write`). If the v3 document is missing the VCEK, the connector consults Tinfoil's ATC service (`POST /attestation` with `{enclaveUrl, repo}`) over a side channel for backfill. It then verifies AMD VCEK chain (ARK → ASK → VCEK, RSA-PSS SHA-384), SEV-SNP / TDX report signature, TCB policy (bl≥0x07, snp≥0x0e, ucode≥0x48), measurement against `ALLOWED_MEASUREMENTS` in `measurements.rs`, and the cert↔report_data binding (`report_data[0..32] == sha256(SPKI(peer_cert))`). Only if all of that passes is the connection yielded to hyper for the real request. ALPN is pinned to `http/1.1` so the inline attestation request and the subsequent application request share one HTTP lifecycle. Subsequent HTTP requests on a pooled keepalive connection do not re-trigger the connector and inherit the binding to the TLS key that was attested when the connection was first established. The same-connection guarantee makes this safe behind load balancers: whatever backend the LB routes you to is the backend you attest.

The enclave's own `/.well-known/tinfoil-attestation?v=3` document is the source of truth. ATC is the single fallback target for the chain itself, and v2 attestation documents are no longer supported. The verifier *does* fetch AMD KDS CRLs in production mode for revocation checks, but this is gated on `trusted_ark_der.is_none()` — test deployments that supply a custom mock ARK skip CRL fetching entirely (AMD KDS has no revocation entries for mock chips and the CRL signature would fail to verify against the mock ARK anyway). `trusted_ark_der` / `trusted_ask_der` only feed the SEV-SNP attestation chain verifier; they are **not** added to any TLS root store.

TLS trust roots are supplied by the caller via `AttestingClientConfig::tls_roots` and used for **all** outbound HTTPS the verifier performs (attested inference endpoint, ATC fallback, AMD KDS CRL fetches). The verifier crate intentionally does not depend on `webpki-roots` or `rustls-native-certs` so each consumer can pick the source that fits its environment without dragging the wrong dep into the others: the server runs `FROM scratch` inside an enclave with no system trust store and supplies `webpki-roots`; the CLI and macOS app supply `rustls-native-certs` so developers can install local dev CAs (e.g. the tinfoil shim mock's `tls-ca.pem`) in their OS keychain. The verifier crate's pure-Rust deps (`sev`, `x509-cert`, `der`, `tower`, `hyper`, `hyper-util`, `httparse`) — no OpenSSL.

This still relies on Tinfoil's TLS private key being sealed inside the enclave: a static attestation document does not defeat an attacker who has somehow exfiltrated the long-lived TLS key. Closing that gap requires per-handshake nonces in `report_data`, which Tinfoil is adding upstream. Once that lands and v3 documents become fully self-contained, the ATC fallback path will go cold and can be removed entirely.

**Compose files:**
- `compose.yaml` — local development. Two supported workflows share one file:
  - **Full container stack** (`just dev`, → `scripts/dev.sh --container`) — postgres + server + shim + stripe-cli all in containers, detached. Server image is rebuilt each invocation.
  - **Host mode** (`just services`, → `scripts/dev.sh --host`) — postgres + shim + stripe-cli in containers, with `SHIM_UPSTREAM_URL=http://host.docker.internal:8080` so the shim forwards to a cargo-built server running on the host. Writes `.env.local` (`STRIPE_WEBHOOK_SECRET` + `BIND_ADDR=0.0.0.0:8080`) for the host server to source.
  - Both modes share `scripts/dev.sh`. Both build only the images they need, idempotently apply `schema.sql` to postgres, capture the Stripe webhook secret if `STRIPE_API_KEY` is set (otherwise skip stripe-cli), and start everything detached. `just down` tears down both modes.
  - Profiles: `server` gates the eidola-server container; `stripe` gates stripe-cli. Postgres and shim have no profile (always available). The shim has `extra_hosts: host.docker.internal:host-gateway` so the host-gateway alias works on Linux too, and intentionally has no `depends_on: server` so it starts cleanly in host mode. `postgres`, `server`, and `shim` declare `platform: linux/amd64` so compose doesn't warn on arm64 hosts about the (intentional) amd64 base layers.

## App Core Architecture

The macOS app, GUI app, and CLI share a common Rust core (`crates/eidola-app-core/`) exposed to Swift via direct [UniFFI](https://mozilla.github.io/uniffi-rs/) bindings and to other Rust callers as a normal library. All business logic — config management, local database, HTTP client construction, account operations, wallet/credential management, and chat inference — lives in the core crate. The CLI (`apps/cli/`) and the gpui-based GUI (`apps/gui/`) are thin Rust wrappers that construct an `AppCore` and call its methods directly; the SwiftUI macOS app (`apps/macos/`) uses the same `AppCore` via UniFFI-generated Swift bindings.

Rust functions and types are exported with `#[uniffi::export]`, `#[derive(uniffi::Object)]`, `#[derive(uniffi::Record)]`, and `#[derive(uniffi::Enum)]`. Async operations use `#[uniffi::export(async)]` to bridge Rust futures to Swift async/await. No serialization layer, event/effect pattern, or Crux dependency — Swift calls Rust functions directly and gets native Swift types back.

**Core crate modules:**
- `lib.rs` — `AppCore` object (UniFFI-exported), all high-level operations (account create/show/allocate, chat, wallet), UniFFI record types (`ConfigState`, `ChatResult`, `PriceInfo`, etc.), internal helpers (ACT token serialization, attestation flushing, HTTP response handling)
- `config.rs` — `Config` struct (TOML serde), load/save with explicit paths, measurement parsing, certificate parsing, domain separator constants
- `db.rs` — Turso (libSQL) database layer with 3-layer schema (wallet, transport, semantic), migrations, all CRUD operations
- `error.rs` — `AppError` enum (UniFFI-exported), request error classification (attestation vs network vs server)

**CLI usage:** The CLI depends on `eidola-app-core` as a regular Rust crate dependency. It calls `AppCore::new(config_dir, data_dir)` and invokes methods directly — no FFI involved.

**macOS app usage:** The macOS app depends on the UniFFI-generated Swift package. `Core.swift` wraps `AppCore` in an `@Observable @MainActor` class that bridges async Rust calls to SwiftUI state. Views: `ChatView` (message bubbles, model picker), `AccountView` (balances, allocation, prices), `WalletView` (credential list), `SettingsView` (base URL, credentials, attestation config).

**GUI app usage (`apps/gui/`):** The gpui-based app depends on `eidola-app-core` as a regular Rust crate. `core.rs` wraps `AppCore` in a gpui `Entity<Core>` that holds cached snapshots (`config_state`, `balances`, `prices`, `credentials`, `models`) and reactively re-renders any view holding the entity via `cx.notify()`. `Core::inner` is `Option<Arc<AppCore>>` — production constructs the core via `Core::new(cx)` (real backend); snapshot tests construct it via `Core::stub()` (no backend, all async methods become no-ops, fields mutated directly to set up render-only fixtures). Async operations are bridged with `tokio::sync::oneshot` channels: the call is `spawn`ed on `AppCore::runtime()` (tokio multi-thread) and awaited from gpui's executor (smol-based) — `oneshot::Receiver` is runtime-agnostic so this is safe. Each view subscribes to `Core` via `cx.observe(&core, ...)`. Native macOS UX is wired with `cx.set_menus(...)` (Eidola → About / Settings… / Hide / Hide Others / Show All / Quit; File → New Space / Close Window; Edit → Undo / Redo / Cut / Copy / Paste / Select All using `gpui_component::input::*`, with Cut/Copy/Paste/Select All declared via `MenuItem::os_action(_, _, OsAction::*)` so they bind to the standard macOS selectors `cut:`/`copy:`/`paste:`/`selectAll:` and route through the responder chain to whatever has focus; Window → Minimize / Zoom), `cx.set_dock_menu(...)` for "New Space" on dock right-click, `cx.bind_keys(...)` (cmd+, cmd+n cmd+w cmd+q cmd+h alt+cmd+h cmd+m + cmd+enter for `Send` in the `ChatView` key-context), and `cx.on_action(...)` handlers. **The "Window" menu name is special** — gpui_macos registers it via `app.setWindowsMenu_(menu)`, which is how AppKit recognizes the app as a fully-wired macOS app and reliably dispatches menu key-equivalents in edge cases (no key window after ⌘Tab back; all windows closed). The Hide/Hide Others/Show All trio are macOS App-menu standards that signal completeness for the same reason. Window-targeting handlers (`CloseWindow`, `Minimize`, `Zoom`) capture `cx.active_window()` and call `cx.defer` to invoke `window.remove_window()` / `minimize_window()` / `zoom_window()` *after* the current update completes — without `defer`, a direct `handle.update(cx, ...)` on the same window we were just dispatched inside fails (its slot is already taken), `.ok()` swallows the Err, and nothing happens. `cx.activate(true)` at launch and `Application::on_reopen` (registered on the builder before `run()`, opens a new chat window when the dock is clicked with none open) round out the standard macOS lifecycle so the app isn't dead-ended after closing the last window. **Window model:** chat windows are non-singleton — every `NewSpace` invocation opens a fresh `ChatView`, each owning its own `space_id` so they're independent conversations sharing the same `Core`. `open_main_window` calls `cx.activate(true)` after `cx.open_window` so a window opened from another app's context (e.g. the dock right-click menu while a different app is foreground) brings Eidola to the front rather than opening behind. **`CloseWindow` is registered per-view** (via `.on_action(cx.listener(…))` in `chat::ChatView` and `settings::SettingsView`, which both `track_focus` a handle that's `focus()`ed in their constructors so the dispatch path reaches the listener even before the user clicks anything), *not* globally. The intentional consequence: `is_action_available` returns true only when a window with the listener is alive, so macOS auto-disables the "Close Window" menu item (and its ⌘W shortcut) when no window is open. The Settings window is a singleton — `AppGlobal.settings_window: Option<WindowHandle<Root>>` caches the handle, and `OpenSettings` raises the existing window via `window.activate_window()` if it's still open. Both open paths are **synchronous** (via `App::open_window`) so the cache is populated before the handler returns. Liveness is checked by matching the cached `WindowId` against `cx.windows()` (the authoritative live list) — borrowing Zed's pattern, except Zed can use `AnyWindowHandle::downcast::<SettingsWindow>` directly because their settings root is uniquely typed; ours is `gpui_component::Root` (shared with chat windows, required for `Input` focus tracking via `Root::read`), so we match by id instead. A stale id self-heals on the next invocation — no `on_release` bookkeeping needed. Views: `chat.rs` (main window), `settings.rs` (custom-tabbed window holding `general.rs`, `account.rs`, `wallet.rs`). Theme is **Circadian** (`theme.rs`) with two `ThemeConfig`s — "Circadian Day" (Light) and "Circadian Night" (Dark) — installed onto the global `gpui_component::Theme` after `gpui_component::init` and applied per OS appearance via `Theme::sync_system_appearance`; each opened window subscribes to appearance changes so toggling macOS Light/Dark updates live. Body font is **Newsreader** (variable TTF, SIL OFL 1.1) bundled at `apps/gui/assets/fonts/` and embedded into the binary via `include_bytes!`, then registered with `cx.text_system().add_fonts`. The crate has both `[lib]` and `[[bin]]` so the snapshot-test integration test can import view modules. Today macOS-only; Linux is the next target. Non-Rust deps introduced: only the system frameworks gpui already pulls in (Cocoa, AppKit, CoreFoundation, CoreGraphics, CoreText, CoreVideo, Metal, Foundation) — no GTK/Qt/node/python. Build deps require Xcode Command Line Tools (`xcode-select --install`).

**GUI testing model.** Two tiers, with different responsibilities:

1. **Behavior tests (`apps/gui/tests/behavior.rs`)** — the regression gate. Built on `gpui::TestAppContext` (mocked rendering, deterministic dispatcher) so they run on libtest's worker thread without AppKit. Pattern: build a `Core::stub()` entity with fixture state, open a window via `cx.open_window`, drive interactions through the view's `focus_handle()` (the same path keystrokes take), assert against public state with `read_with`. Stub cores have `inner: None`, so `Core::app_core()` returns `None`; views that hit that path early-return after the local state mutation (e.g. `ChatView::submit` pushes the user message and sets `thinking=true` then bails), which is enough to test the parts that don't need a backend. HTTP-mocked tests (real `AppCore` against a `wiremock` server) are the natural next layer.

2. **Visual snapshots (`apps/gui/tests/visual.rs`)** — local-only debug aid, **not** a regression gate. Built on `gpui::VisualTestAppContext` (real Metal renderer, offscreen window at -10000,-10000, deterministic dispatcher). Configured as `[[test]] harness = false` so `fn main()` runs on the macOS main thread (libtest's worker-thread harness would SIGABRT inside AppKit). Cases live in `tests/visual/cases.rs`; the harness in `tests/visual/harness.rs` wraps each user view in a `Root` and renders it **twice — once in Circadian Day (Light), once in Circadian Night (Dark)** — by calling `Theme::change` between renders. Each case writes/compares two files: `tests/snapshots/<name>-day.png` and `<name>-night.png`. Case build closures must be `Fn` (invoked once per mode); they construct fresh entities each call. The PNGs are **gitignored** — pixels are platform- and machine-bound (Metal+CoreText vs wgpu+cosmic-text on Linux; font hinting differs across macOS minor versions), so committing them would mean false-positive regressions in CI and on every other developer's machine. Their value is local: agents/humans can `Read` a PNG to "feel" a view at a state, and a developer iterating on a UI change can re-render and eyeball-diff their previous run. Behavior: missing PNG → write it and report `written`; mismatch against a previously-written local PNG → write `<name>-<mode>.new.png` for review and fail; `UPDATE_SNAPSHOTS=1` overwrites. Recipes: `just render-snapshots` (verify/write) and `just render-snapshots-update` (accept). `cargo test -p eidola-gui` runs both tiers — the visual tier prints `written` on a fresh checkout and `ok` on subsequent runs against the locally-cached PNGs, never gating CI.

**Why both tiers?** Behavior tests catch logic regressions (clicking X must call `core.Y(z)`; an empty Send must be a no-op) and survive across platforms. Visual snapshots are the "did I accidentally change the layout?" check that's only meaningful to the dev making the change. Together they let agents make UI changes confidently: behavior tests gate the merge, visual snapshots let the agent verify the change *looks* right by reading the freshly-written PNG.

**Crate layout:** Pure Rust crates in `crates/` implement capability logic. The `crates/` tree also contains the Rust code generation binary (`uniffi-bindgen-swift`) plus operational utilities such as `generate-openapi`, `tinfoil-shim-mock`, `hash-secret`, and `measure-enclave`.

**Codegen pipeline:**
- `uniffi-bindgen-swift` (workspace crate under `crates/`) → FFI bridge (Swift bindings + C headers)

## Local Database & Migrations

Both the CLI and macOS app use an embedded [Turso](https://crates.io/crates/turso) (pure-Rust libSQL) database at `~/Library/Application Support/eidola/eidola.db` for local app data (wallet credentials, conversation history, attestation records, etc.). The database layer lives in `crates/eidola-app-core/src/db.rs`.

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

**Prerequisites:** `rustup`, `just`, `docker`

The `justfile` is the primary development interface. Run `just` to see all available recipes.

**Key recipes:**
- `just build {server,cli,gui,macos}` — local-toolchain builds for fast iteration. The `macos` target regenerates UniFFI bindings and the XCFramework, runs `swift build`, then assembles a `.app` bundle via `scripts/package-macos-app.sh` into `apps/macos/build/Eidola.app`. The `gui` target on macOS additionally runs `scripts/package-gui-app.sh` to assemble `apps/gui/build/Eidola.app` (Info.plist at `apps/gui/Support/Info.plist`, ad-hoc codesigned). The .app wrapper is **required**, not cosmetic: AppKit treats a bare `cargo run` binary as a command-line tool rather than a real app, which breaks menu key-equivalent dispatch (⌘N / ⌘Q / ⌘,) when no window has key focus — the diagnostic signal is the menu bar showing the binary name (`eidola-gui`) instead of the app name (`Eidola`).
- `just run {server,cli,gui,macos}` — build and run. For `macos` and `gui`, opens the assembled `.app` via `open`. Accepts trailing args (e.g. `just run cli chat "hello"`).
- `just test` — runs `cargo test` plus Swift tests (`crates/eidola-app-core` and `apps/macos`) on macOS.
- `just check` — clippy, rustfmt, swift-format lint.
- `just dev` / `just services` / `just down` — container-based development workflows (see Compose files above).

## Conventions

- Pure Rust dependencies preferred (for cross-compilation)
- Keep Rust workspace packages under `crates/`; do not add a separate top-level `tools/` tree
- `just` is the task runner — wrap scripts and common commands as recipes
- Server and CLI OCI images are built with StageX (reproducible, `FROM scratch`, runs as non-root)
- Nix is used for CI quality gates and Swift/XCFramework builds, not daily Rust development
- `rustup` + `rust-toolchain.toml` manages the Rust toolchain for development
- OpenAI API format as the canonical interface
- Server API is documented via utoipa `#[utoipa::path]` annotations on handler functions and `ToSchema` derives on request/response types. `OpenApiRouter` (in `lib.rs::build_router()`) collects paths and recursively discovers schemas automatically — only SSE streaming types that aren't referenced from path annotations are listed manually in `api_doc.rs`. When adding or changing server endpoints, add the annotation on the handler and register it in `build_router()` via `routes!()`, then run `just update-openapi` to regenerate the committed `openapi.json`
- `artifact-manifest.json` (v1 format) records expected OCI digests, macOS app/CLI Nix `narHash` values, and enclave measurements (SEV-SNP + TDX + cmdline) with type/platform metadata; the enclave block shape matches the Tinfoil `snp-tdx-multiplatform/v1` predicate so `tinfoil-build.yml` can project it directly into `tinfoil-deployment.json`. CI verifies the full file by merging digests captured from the real OCI and macOS build jobs and recomputing enclave measurements from `tinfoil-config.yml`. Use `just update-manifest` to regenerate it on macOS with the pinned amd64 BuildKit builder plus the local Nix macOS builds
- Contributor agreement state lives in `CLA-INDIVIDUAL.md`, `CLA-CORPORATE.md`, and `CLA-SIGNERS.txt`; the signer ledger plus Git history is the source of truth, and changing either CLA text requires new signer entries because the SHA-256 hash changes
- Before committing, ensure `README.md` and `AGENTS.md` are updated to reflect any changes (new files, endpoints, env vars, build commands, etc.)
- Omit any tool-specific "co-authored by" lines from commit messages
