//! Local model management — download, storage, and llama.cpp engine
//! lifecycle for on-device inference.
//!
//! ## Design
//!
//! Local inference runs through **llama.cpp in server mode**: a managed
//! `llama-server` subprocess per loaded model, speaking the same
//! OpenAI-compatible HTTP the remote Eidola server speaks, bound to
//! `127.0.0.1` on an ephemeral port. The chat path treats a loaded local
//! model as just another upstream — `prepare_turn` swaps the attested
//! client + credential spend for a plain loopback client with no billing,
//! and every downstream mechanism (SSE parsing, context assembly, the
//! durable turn rows) is shared with the remote path.
//!
//! Process isolation is deliberate: an inference crash (OOM, driver bugs)
//! kills the child, never the app, and the subprocess boundary is the same
//! OpenAI-HTTP seam a self-hosted second-device server would plug into.
//!
//! Two backend kinds share this module's machinery (see `crate::backends`):
//! the managed **`local`** singleton (Eidola's own model store under
//! `<data_dir>/models/` — downloads, catalog, delete) and any number of
//! **`llamacpp`** backends (a *user-owned* llama.cpp install: Eidola scans
//! the backend's `models_dir` and starts/stops engines, but never
//! downloads or deletes the files). Engines are keyed by
//! `(backend_id, slug)`.
//!
//! ## State
//!
//! Durable truth is the filesystem: one `<file>.gguf` per model plus (in
//! the managed store) a `<file>.gguf.meta.json` sidecar (display name,
//! source URL, download time). A `.gguf` dropped in manually is picked up
//! on the next scan — the sidecar is optional. In-flight downloads write
//! `<file>.gguf.part` and rename on completion, so a crash never leaves a
//! truncated file masquerading as a model.
//!
//! Runtime state (downloads in progress, running engines) lives in
//! [`LocalRuntime`] on `Inner`. Every state transition emits
//! [`Change::LocalModels`] so subscribers re-snapshot via
//! `AppCore::local_models_state`; download progress emits throttled.
//! Model ids are the uniform qualified form — `<file-stem>@<backend-id>`,
//! e.g. `<file-stem>@local` for the managed store — and the id doubles as
//! the spawned engine's `--alias`, so a chat body's `model` field matches
//! the selection string verbatim.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::Inner;
use crate::changes::{BroadcastSource, Change};
use crate::config::Config;
use crate::error::AppError;

/// Context window requested from `llama-server` (`-c`). Deliberately
/// bounded — KV-cache memory scales with it — and far below what Gemma 4
/// supports; a later change can make this per-model configuration.
pub(crate) const LOCAL_CONTEXT_TOKENS: u32 = 8192;

/// How long a spawned engine gets to reach `/health` = 200 before the load
/// is declared failed. Large models on a cold page cache legitimately take
/// minutes to mmap and compile pipelines.
const ENGINE_READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Poll interval against `/health` while the engine is loading.
const ENGINE_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// Minimum interval between download-progress bus emissions.
const PROGRESS_EMIT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(400);

// ============================================================================
// Curated catalog — the official Google Gemma 4 QAT GGUF releases.
// ============================================================================

/// One curated downloadable model. `file_name` is the exact object name on
/// Hugging Face; installed-state matching keys off it.
#[derive(Clone, Copy, Debug)]
pub struct LocalCatalogEntry {
    pub id: &'static str,
    pub display_name: &'static str,
    pub description: &'static str,
    pub url: &'static str,
    pub file_name: &'static str,
    pub size_bytes: u64,
}

/// Official Gemma 4 instruction-tuned QAT Q4_0 GGUFs published by Google.
/// Sizes are the exact object sizes reported by the Hugging Face tree API.
pub const LOCAL_MODEL_CATALOG: &[LocalCatalogEntry] = &[
    LocalCatalogEntry {
        id: "gemma-4-e2b",
        display_name: "Gemma 4 E2B",
        description: "Fastest — ~2B effective parameters; light enough for any Apple-silicon Mac",
        url: "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf",
        file_name: "gemma-4-E2B_q4_0-it.gguf",
        size_bytes: 3_349_514_112,
    },
    LocalCatalogEntry {
        id: "gemma-4-e4b",
        display_name: "Gemma 4 E4B",
        description: "Balanced — ~4B effective parameters; the everyday choice on 16 GB machines",
        url: "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf",
        file_name: "gemma-4-E4B_q4_0-it.gguf",
        size_bytes: 5_154_939_136,
    },
    LocalCatalogEntry {
        id: "gemma-4-12b",
        display_name: "Gemma 4 12B",
        description: "Strong general model; wants 16 GB+ of memory",
        url: "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/gemma-4-12b-it-qat-q4_0.gguf",
        file_name: "gemma-4-12b-it-qat-q4_0.gguf",
        size_bytes: 6_975_877_728,
    },
    LocalCatalogEntry {
        id: "gemma-4-26b-a4b",
        display_name: "Gemma 4 26B (A4B)",
        description: "Mixture-of-experts — 26B total, 4B active per token; fast for its quality, wants 32 GB+",
        url: "https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/main/gemma-4-26B_q4_0-it.gguf",
        file_name: "gemma-4-26B_q4_0-it.gguf",
        size_bytes: 14_439_361_440,
    },
    LocalCatalogEntry {
        id: "gemma-4-31b",
        display_name: "Gemma 4 31B",
        description: "The flagship dense Gemma 4; wants 32 GB+ of memory",
        url: "https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf/resolve/main/gemma-4-31B_q4_0-it.gguf",
        file_name: "gemma-4-31B_q4_0-it.gguf",
        size_bytes: 17_650_999_456,
    },
];

// ============================================================================
// DTOs — snapshots returned to the CLI/GUI.
// ============================================================================

/// Lifecycle state of one local model, merged from the filesystem and the
/// runtime maps.
#[derive(Clone, Debug, PartialEq)]
pub enum LocalModelStatus {
    /// A download task is streaming this model to disk.
    Downloading { received: u64, total: Option<u64> },
    /// On disk, no engine running.
    Available,
    /// An engine subprocess is spawned and warming up (polling `/health`).
    Loading,
    /// Serving on `127.0.0.1:<port>`. `pinned` engines are protected from
    /// automatic (LRU) unloading; manual unload still applies.
    Loaded {
        port: u16,
        context_tokens: u32,
        pinned: bool,
    },
}

/// One engine-served model as shown in Settings → Backends and (when
/// loaded) the model picker.
#[derive(Clone, Debug)]
pub struct LocalModelInfo {
    /// The chat-routable selection id: `<slug>@<backend-id>` (the managed
    /// store's backend id is `local`).
    pub id: String,
    /// The file stem.
    pub slug: String,
    pub display_name: String,
    pub file_name: String,
    /// On-disk size (or bytes expected while downloading, if known).
    pub size_bytes: Option<u64>,
    pub source_url: Option<String>,
    pub status: LocalModelStatus,
    /// The most recent download/load failure for this slug, until a retry
    /// replaces it. Surfaced so failures are visible, never silent.
    pub last_error: Option<String>,
}

/// Snapshot of the whole local-inference domain: the managed `local`
/// singleton plus every configured `llamacpp` backend's scanned directory.
#[derive(Clone, Debug)]
pub struct LocalModelsState {
    /// The resolved **bundled** `local` engine (config override,
    /// `EIDOLA_LLAMA_SERVER`, or the exe-relative sidecar). `None` ⇒ this
    /// build ships no engine — the UI shows an honest "engine not present"
    /// state and the `llama_server_path` override is the escape hatch.
    pub engine_path: Option<String>,
    /// The `local` singleton's models (Eidola-managed store).
    pub models: Vec<LocalModelInfo>,
    /// Each configured `llamacpp` backend: its user-owned directory scanned
    /// for `.gguf`s, with live engine status. Eidola never downloads or
    /// deletes here — the verbs are load/unload only.
    pub external: Vec<ExternalEngineBackend>,
}

/// One `llamacpp` backend's scanned models.
#[derive(Clone, Debug)]
pub struct ExternalEngineBackend {
    pub backend_id: String,
    pub display_name: String,
    pub enabled: bool,
    pub models_dir: String,
    /// The resolved `llama-server` for this backend (its explicit
    /// `engine_path`, else discovery), if one was found.
    pub engine_path: Option<String>,
    /// Whether a request may auto-start an engine for this backend.
    pub auto_start: bool,
    pub models: Vec<LocalModelInfo>,
}

/// Sidecar metadata written next to each downloaded `.gguf`.
#[derive(Serialize, Deserialize)]
struct ModelSidecar {
    display_name: String,
    source_url: String,
    downloaded_at_ms: i64,
}

// ============================================================================
// Runtime state
// ============================================================================

/// Live download bookkeeping. Shared with the transfer task via `Arc` so
/// the snapshot reads progress without any channel plumbing.
struct DownloadEntry {
    display_name: String,
    source_url: String,
    received: AtomicU64,
    /// 0 = unknown (no Content-Length yet).
    total: AtomicU64,
    cancel: AtomicBool,
}

/// A running (or warming-up) engine subprocess. The supervisor task owns
/// the child process; this entry is the control handle.
struct EngineEntry {
    port: u16,
    context_tokens: u32,
    ready: bool,
    /// Consumed by `unload` (map removal hands us ownership); the
    /// supervisor kills the child on receipt or on drop.
    shutdown: tokio::sync::oneshot::Sender<()>,
    /// Estimated resident footprint (file size + [`ENGINE_OVERHEAD_BYTES`])
    /// — the eviction planner's currency.
    footprint: u64,
    /// Pinned engines are protected from *automatic* (LRU) unloading;
    /// manual unload still applies. Runtime state — set from Settings →
    /// Backends on a loaded engine, gone with it.
    pinned: bool,
    /// When a turn last leased this engine (ms since epoch; load time
    /// initially) — the LRU clock.
    last_used_ms: i64,
    /// Turns currently running against this engine. `Arc` so an
    /// [`EngineLease`]'s decrement survives the entry being removed.
    in_flight: Arc<AtomicU64>,
}

/// An in-flight-turn hold on an engine: taken when a turn routes to it,
/// released (`Drop`) when the turn ends — success or failure. While any
/// lease is live the engine is never auto-unloaded.
pub(crate) struct EngineLease {
    in_flight: Arc<AtomicU64>,
}

impl Drop for EngineLease {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Rolls back a warming-engine reservation (made by
/// [`LocalRuntime::reserve_engine`]) if the load never reaches its
/// supervisor — an error return or the loading future being dropped
/// between reserve and spawn. Without it a stranded warming entry would
/// wedge every later load of the model into joining a load no task owns.
struct ReservationGuard {
    local: Arc<LocalRuntime>,
    bus: BroadcastSource,
    key: Option<EngineKey>,
}

impl ReservationGuard {
    /// The supervisor now owns the entry; the guard stands down.
    fn defuse(mut self) {
        self.key = None;
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if let Some(key) = self.key.take() {
            self.local
                .engines
                .lock()
                .expect("engines lock")
                .remove(&key);
            self.bus.emit(Change::LocalModels);
        }
    }
}

/// Engine/failure map key: `(backend_id, slug)`. The `local` singleton and
/// every `llamacpp` backend share one runtime, so the backend id is part of
/// the identity (two backends may hold same-named files).
pub(crate) type EngineKey = (String, String);

/// All runtime (non-durable) local-inference state, held by `Inner` behind
/// an `Arc` so transfer/supervisor tasks can outlive individual calls.
/// Plain `std::sync::Mutex` — never held across an `.await`.
#[derive(Default)]
pub(crate) struct LocalRuntime {
    /// Downloads exist only for the managed `local` backend, keyed by slug.
    downloads: StdMutex<HashMap<String, Arc<DownloadEntry>>>,
    engines: StdMutex<HashMap<EngineKey, EngineEntry>>,
    /// Last failure per (backend, slug) — download or load — until retried.
    failures: StdMutex<HashMap<EngineKey, String>>,
    /// Test override for the machine memory budget ([`memory_budget`]).
    budget_override: StdMutex<Option<u64>>,
    /// One-way shutdown latch. Set by [`Inner::shutdown_all_engines`] and
    /// checked by [`Self::reserve_engine`] — **both under the `engines`
    /// lock**, which is what makes it race-free; the atomic is only here
    /// because the lock guards a bare `HashMap` rather than a state struct.
    ///
    /// Without it, a load already past its `await`s (backend lookup, port
    /// pick, `fs::metadata`) but not yet at its reservation would resume
    /// *after* the drain observed an empty registry, reserve, and spawn a
    /// subprocess that the imminent `exit()` orphans. Never cleared: the
    /// process is quitting.
    shutting_down: std::sync::atomic::AtomicBool,
}

impl LocalRuntime {
    /// The loopback base URL + context window of a ready engine, if that
    /// backend's slug is loaded. `None` while still warming up.
    pub(crate) fn ready_engine(&self, backend_id: &str, slug: &str) -> Option<(String, u32)> {
        let engines = self.engines.lock().expect("engines lock");
        engines
            .get(&(backend_id.to_string(), slug.to_string()))
            .and_then(|e| {
                e.ready
                    .then(|| (format!("http://127.0.0.1:{}", e.port), e.context_tokens))
            })
    }

    /// Lease a ready engine for one turn: bumps the LRU clock and the
    /// in-flight count (released when the returned [`EngineLease`] drops).
    /// `None` while absent or still warming up.
    pub(crate) fn lease_engine(
        &self,
        backend_id: &str,
        slug: &str,
    ) -> Option<(String, u32, EngineLease)> {
        let mut engines = self.engines.lock().expect("engines lock");
        let entry = engines.get_mut(&(backend_id.to_string(), slug.to_string()))?;
        if !entry.ready {
            return None;
        }
        entry.last_used_ms = crate::now_ms();
        entry.in_flight.fetch_add(1, Ordering::SeqCst);
        Some((
            format!("http://127.0.0.1:{}", entry.port),
            entry.context_tokens,
            EngineLease {
                in_flight: entry.in_flight.clone(),
            },
        ))
    }

    /// Whether an entry (ready or warming) exists for this key.
    fn engine_present(&self, key: &EngineKey) -> bool {
        self.engines.lock().expect("engines lock").contains_key(key)
    }

    /// Atomically plan evictions for a new engine and insert its warming
    /// entry — one critical section over the engines map, so two concurrent
    /// loads of different models can never both conclude that the same free
    /// memory covers them (each sees the other's reservation in its own
    /// plan). Returns `Ok(None)` when an entry for `key` already exists
    /// (join that load) and `Ok(Some(victims))` when the reservation is in
    /// place; `Err` refuses without reserving or unloading anything.
    fn reserve_engine(
        &self,
        key: &EngineKey,
        port: u16,
        footprint: u64,
        shutdown: tokio::sync::oneshot::Sender<()>,
    ) -> Result<Option<Vec<EngineKey>>, String> {
        let mut engines = self.engines.lock().expect("engines lock");
        // The shutdown authority is checked *here*, inside the reservation's
        // critical section and under the same lock the drain takes — not by
        // the caller before its `await`s, which is exactly the window that
        // would let a mid-flight load spawn a subprocess after the drain.
        if self.shutting_down.load(Ordering::SeqCst) {
            return Err("Eidola is shutting down".into());
        }
        if engines.contains_key(key) {
            return Ok(None);
        }
        let snapshot: Vec<EngineUsage> = engines
            .iter()
            .map(|(key, e)| EngineUsage {
                key: key.clone(),
                footprint: e.footprint,
                last_used_ms: e.last_used_ms,
                pinned: e.pinned,
                in_flight: e.in_flight.load(Ordering::SeqCst),
                warming: !e.ready,
            })
            .collect();
        let victims = plan_evictions(footprint, self.memory_budget(), &snapshot)?;
        engines.insert(
            key.clone(),
            EngineEntry {
                port,
                context_tokens: LOCAL_CONTEXT_TOKENS,
                ready: false,
                shutdown,
                footprint,
                pinned: false,
                last_used_ms: crate::now_ms(),
                in_flight: Arc::new(AtomicU64::new(0)),
            },
        );
        Ok(Some(victims))
    }

    /// Run `f` — the engine spawn — only if no quit-time shutdown has begun,
    /// **atomically with respect to the drain**.
    ///
    /// The latch on [`Self::reserve_engine`] closes the window between a load's
    /// last `await` and its reservation; this closes the one *after* it. A
    /// reservation accepted a moment before the latch flipped hands its
    /// supervisor to `tokio::spawn`, and that task's first poll can land after
    /// the drain has already walked an empty-of-it registry — at which point an
    /// unconditional `command.spawn()` starts a subprocess into a process about
    /// to `exit()`. Same class, same cure: the authority is read at the write
    /// point, under the same lock the drain takes, so no interleaving can put a
    /// spawn after the latch.
    ///
    /// Holding the `engines` lock across the spawn is safe and deliberate:
    /// `Command::spawn` is synchronous (no `await` under the guard) and touches
    /// nothing in this map.
    fn spawn_unless_shutting_down<T>(&self, f: impl FnOnce() -> T) -> Option<T> {
        let _engines = self.engines.lock().expect("engines lock");
        if self.shutting_down.load(Ordering::SeqCst) {
            return None;
        }
        Some(f())
    }

    /// The total memory the engine pool may occupy: a fixed fraction of
    /// physical RAM (leaving headroom for the app and the OS), or the test
    /// override. The estimate errs permissive — a genuinely-too-big load
    /// still fails at the engine and surfaces honestly.
    fn memory_budget(&self) -> u64 {
        if let Some(b) = *self.budget_override.lock().expect("budget lock") {
            return b;
        }
        total_memory_bytes()
            .map(|total| total / MEMORY_BUDGET_DEN * MEMORY_BUDGET_NUM)
            // Unknown RAM (exotic platform): a permissive fallback — the
            // planner never evicts needlessly and real failures surface.
            .unwrap_or(u64::MAX / 2)
    }

    /// Test seam: pin the memory budget so eviction tests are
    /// deterministic on any machine.
    #[doc(hidden)]
    pub(crate) fn set_memory_budget_for_test(&self, budget: u64) {
        *self.budget_override.lock().expect("budget lock") = Some(budget);
    }

    /// Test seam: register a fake "ready" engine at an arbitrary port so
    /// integration tests can route local turns at a mock upstream without
    /// spawning a real llama-server.
    #[doc(hidden)]
    pub(crate) fn register_for_test(&self, backend_id: &str, slug: &str, port: u16) {
        self.register_engine_for_test(backend_id, slug, port, ENGINE_OVERHEAD_BYTES, false, 0);
    }

    /// Test seam: like [`register_for_test`] but with explicit footprint,
    /// pin state, and LRU timestamp — the eviction tests' fixture.
    #[doc(hidden)]
    pub(crate) fn register_engine_for_test(
        &self,
        backend_id: &str,
        slug: &str,
        port: u16,
        footprint: u64,
        pinned: bool,
        last_used_ms: i64,
    ) {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        self.engines.lock().expect("engines lock").insert(
            (backend_id.to_string(), slug.to_string()),
            EngineEntry {
                port,
                context_tokens: LOCAL_CONTEXT_TOKENS,
                ready: true,
                shutdown: tx,
                footprint,
                pinned,
                last_used_ms,
                in_flight: Arc::new(AtomicU64::new(0)),
            },
        );
    }
}

// ============================================================================
// Memory budget + the eviction planner
// ============================================================================

/// Fraction of physical RAM the engine pool may occupy (numerator /
/// denominator): 4/5. The remainder is headroom for the app, the OS, and
/// the KV-cache estimate error.
const MEMORY_BUDGET_NUM: u64 = 4;
const MEMORY_BUDGET_DEN: u64 = 5;

/// Estimated per-engine overhead beyond the mmapped weights: KV cache at
/// [`LOCAL_CONTEXT_TOKENS`], compute buffers, and the process itself. A
/// deliberate rough heuristic — the planner only decides *evictions*; a
/// load that genuinely doesn't fit still fails at the engine and surfaces
/// as an honest error.
const ENGINE_OVERHEAD_BYTES: u64 = 1 << 30; // 1 GiB

/// Estimated resident footprint of an engine serving a model file.
fn engine_footprint(file_size: u64) -> u64 {
    file_size.saturating_add(ENGINE_OVERHEAD_BYTES)
}

/// Physical RAM, if the platform tells us.
fn total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let mut size: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut size as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(size)
    }
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = meminfo
            .lines()
            .find(|l| l.starts_with("MemTotal:"))?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()?;
        Some(kb * 1024)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// One loaded engine as the eviction planner sees it.
#[derive(Clone, Debug)]
struct EngineUsage {
    key: EngineKey,
    footprint: u64,
    last_used_ms: i64,
    pinned: bool,
    in_flight: u64,
    /// Still warming up (another load's reservation) — its memory is
    /// committed but there is no engine to gracefully unload yet.
    warming: bool,
}

/// Decide which engines to unload so a new engine of `required` footprint
/// fits inside `budget`. Pure, so the policy is unit-testable:
///
/// - Nothing to do when it already fits.
/// - Otherwise evict least-recently-used first, but **only** engines that
///   are neither pinned nor serving an in-flight turn.
/// - If even evicting every candidate can't make room, `Err` explains
///   what's holding the memory — the caller surfaces it without unloading
///   anything (a pointless eviction would punish the user twice).
fn plan_evictions(
    required: u64,
    budget: u64,
    engines: &[EngineUsage],
) -> Result<Vec<EngineKey>, String> {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    if required > budget {
        return Err(format!(
            "the model needs ~{:.1} GiB but the memory budget is {:.1} GiB",
            gib(required),
            gib(budget)
        ));
    }
    let in_use: u64 = engines.iter().map(|e| e.footprint).sum();
    let mut free = budget.saturating_sub(in_use);
    if free >= required {
        return Ok(Vec::new());
    }

    let mut candidates: Vec<&EngineUsage> = engines
        .iter()
        .filter(|e| !e.pinned && e.in_flight == 0 && !e.warming)
        .collect();
    candidates.sort_by_key(|e| e.last_used_ms);

    let mut evict = Vec::new();
    for e in candidates {
        if free >= required {
            break;
        }
        free = free.saturating_add(e.footprint);
        evict.push(e.key.clone());
    }
    if free >= required {
        Ok(evict)
    } else {
        let held: u64 = engines
            .iter()
            .filter(|e| e.pinned || e.in_flight > 0 || e.warming)
            .map(|e| e.footprint)
            .sum();
        Err(format!(
            "the model needs ~{:.1} GiB, but ~{:.1} GiB is held by pinned or in-use models — \
             unpin or unload one in Settings → Backends",
            gib(required),
            gib(held)
        ))
    }
}

// ============================================================================
// Pure helpers
// ============================================================================

/// Directory holding downloaded model files.
pub(crate) fn models_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("models")
}

/// The selectable model id for an engine-backed backend's slug — the
/// uniform qualified form ([`crate::backends::qualified_model_id`]). The id
/// doubles as the engine's `--alias`, so the wire model in a chat body
/// equals the selection string.
pub(crate) fn engine_model_id(backend_id: &str, slug: &str) -> String {
    crate::backends::qualified_model_id(slug, backend_id)
}

/// Resolve a model-management argument (`<slug>@<backend>`, or a bare slug
/// as shorthand for the managed local store — the natural spelling for
/// `eidola model download/load/…`) to its engine key `(backend_id, slug)`.
pub(crate) fn engine_key_for_id(id: &str) -> EngineKey {
    let mref = crate::backends::parse_model_ref(id);
    if mref.backend_id == crate::backends::EIDOLA_BACKEND_ID {
        // Bare parses as eidola (the chat sugar); in this module's
        // management vocabulary a bare slug means the local store.
        (crate::backends::LOCAL_BACKEND_ID.to_string(), mref.model)
    } else {
        (mref.backend_id, mref.model)
    }
}

/// Normalize a pasted model URL into `(download_url, file_name)`.
///
/// Accepts direct `.gguf` URLs and Hugging Face `/blob/` page URLs (the
/// address bar of a file page), which are rewritten to `/resolve/` so they
/// download instead of rendering HTML. Anything without a `.gguf` final
/// path segment is rejected — better an honest error at paste time than a
/// half-downloaded HTML page named like a model.
pub fn normalize_model_url(input: &str) -> Result<(String, String), AppError> {
    let raw = input.trim();
    if raw.is_empty() {
        return Err(AppError::LocalModel {
            message: "enter a model URL".into(),
        });
    }
    if !(raw.starts_with("https://") || raw.starts_with("http://")) {
        return Err(AppError::LocalModel {
            message: "model URL must start with https://".into(),
        });
    }
    // Drop query/fragment (e.g. `?download=true`) before inspecting the path.
    let without_fragment = raw.split('#').next().unwrap_or(raw);
    let path_part = without_fragment
        .split('?')
        .next()
        .unwrap_or(without_fragment);

    // Hugging Face file-page URLs use `/blob/<rev>/...`; the raw object
    // lives at `/resolve/<rev>/...`.
    let url = if path_part.contains("huggingface.co/") {
        path_part.replacen("/blob/", "/resolve/", 1)
    } else {
        path_part.to_string()
    };

    let file_name = url.rsplit('/').next().unwrap_or_default().to_string();
    if !file_name.to_ascii_lowercase().ends_with(".gguf") {
        return Err(AppError::LocalModel {
            message: format!(
                "URL must point to a .gguf file (got `{}`)",
                if file_name.is_empty() {
                    "no file name"
                } else {
                    &file_name
                }
            ),
        });
    }
    if !file_name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
    {
        return Err(AppError::LocalModel {
            message: format!("model file name contains unsupported characters: `{file_name}`"),
        });
    }
    Ok((url, file_name))
}

/// The slug (and thus the `<slug>@local` id) for a model file name. The
/// extension strip is case-insensitive to match the directory scan, which
/// accepts any case variant of `.gguf`.
fn slug_for_file(file_name: &str) -> String {
    match file_name
        .len()
        .checked_sub(5)
        .and_then(|i| file_name.get(i..))
    {
        Some(ext) if ext.eq_ignore_ascii_case(".gguf") => {
            file_name[..file_name.len() - 5].to_string()
        }
        _ => file_name.to_string(),
    }
}

/// Resolve a slug back to its on-disk model file. The scan accepts any case
/// variant of the `.gguf` extension, so on a case-sensitive filesystem the
/// synthesized `<slug>.gguf` may not name the file that was advertised —
/// fall back to scanning the directory for the file whose slug matches.
async fn find_model_file(dir: &Path, slug: &str) -> Option<PathBuf> {
    let exact = dir.join(format!("{slug}.gguf"));
    if exact.is_file() {
        return Some(exact);
    }
    let mut entries = tokio::fs::read_dir(dir).await.ok()?;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let matches = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.to_ascii_lowercase().ends_with(".gguf") && slug_for_file(n) == slug);
        if matches && path.is_file() {
            return Some(path);
        }
    }
    None
}

/// A human display name derived from a file stem when no sidecar or
/// catalog entry supplies one.
fn prettify_stem(stem: &str) -> String {
    stem.replace(['-', '_'], " ")
}

/// Candidate locations for the **bundled** `local` engine, exe-relative, in
/// probe order. Pure over the executable path so it's unit-testable.
///
/// - macOS `.app`: the main binary sits at `…/Contents/MacOS/<x>`; the
///   sidecar is shipped at `…/Contents/Resources/bin/llama-server`.
/// - Otherwise (CLI Nix output, `target/debug` dev builds): a `llama-server`
///   sibling next to the executable.
fn bundled_engine_candidates(exe: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = exe.parent() {
        if dir.file_name() == Some(std::ffi::OsStr::new("MacOS"))
            && let Some(contents) = dir.parent()
        {
            candidates.push(contents.join("Resources").join("bin").join("llama-server"));
        }
        candidates.push(dir.join("llama-server"));
    }
    candidates
}

/// Resolve the **`local`** (bundled) engine, in order: the config override
/// (`llama_server_path`, the dev escape hatch + test seam), the
/// `EIDOLA_LLAMA_SERVER` env var (set by the Linux GUI wrapper), then the
/// exe-relative bundled sidecar. There is **no `$PATH` scan** — the managed
/// engine is shipped with the app, never borrowed from the system. Pure over
/// its injected inputs so the ordering is unit-testable.
fn resolve_local_engine_path(
    override_path: Option<&str>,
    env_path: Option<&str>,
    current_exe: Option<&Path>,
    exists: &dyn Fn(&Path) -> bool,
) -> Option<PathBuf> {
    // The override is a pin: set-but-missing resolves to `None` rather than
    // silently falling through (honest for an explicit escape hatch).
    if let Some(p) = override_path {
        let p = PathBuf::from(p);
        return exists(&p).then_some(p);
    }
    if let Some(p) = env_path {
        let p = PathBuf::from(p);
        if exists(&p) {
            return Some(p);
        }
    }
    if let Some(exe) = current_exe {
        for candidate in bundled_engine_candidates(exe) {
            if exists(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// The `local` (bundled) engine binary, resolved against the live process.
pub(crate) fn resolve_local_engine(cfg: &Config) -> Option<PathBuf> {
    let env = std::env::var("EIDOLA_LLAMA_SERVER").ok();
    let exe = std::env::current_exe().ok();
    resolve_local_engine_path(
        cfg.llama_server_path_override.as_deref(),
        env.as_deref(),
        exe.as_deref(),
        &usable_engine_binary,
    )
}

/// Whether `path` is an engine binary this machine can actually execute.
/// The macOS universal app deliberately ships the arm64-only sidecar (see
/// the workspace `AGENTS.md`), so on an Intel Mac the file *exists* inside
/// the `.app` but can only fail with an exec-format error — treating it as
/// present would advertise an engine instead of the honest missing-engine
/// state. Anything unreadable or unrecognized passes through: the check
/// only rejects the known-dishonest case, and a real spawn failure still
/// surfaces its own error.
fn usable_engine_binary(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        use std::io::Read;
        let mut header = Vec::new();
        let readable = std::fs::File::open(path)
            .map(|f| f.take(MACHO_HEADER_READ_LEN).read_to_end(&mut header))
            .is_ok();
        if readable {
            return macho_machine_compatible(
                macho_cpu_types(&header).as_deref(),
                machine_supports_arm64(),
            );
        }
    }
    true
}

/// Enough bytes for a fat header with a generous arch count (8-byte header
/// + 32 `fat_arch_64` entries at 32 bytes each).
#[cfg(target_os = "macos")]
const MACHO_HEADER_READ_LEN: u64 = 8 + 32 * 32;

#[cfg(any(test, target_os = "macos"))]
const CPU_TYPE_ARM64: u32 = 0x0100_000C;

/// Whether the current machine can execute arm64 code (`hw.optional.arm64`
/// — present and 1 on Apple Silicon, including under Rosetta; absent on
/// Intel Macs).
#[cfg(target_os = "macos")]
fn machine_supports_arm64() -> bool {
    let mut val: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    let name = c"hw.optional.arm64";
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut val as *mut i32 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    rc == 0 && val == 1
}

/// Pure policy over a parsed Mach-O header: reject only a file whose every
/// slice is arm64 on a machine that can't execute arm64. Unknown formats
/// (`None`/empty) pass — the spawn surfaces real errors.
#[cfg(any(test, target_os = "macos"))]
fn macho_machine_compatible(cpu_types: Option<&[u32]>, machine_arm64: bool) -> bool {
    match cpu_types {
        Some(types) if !types.is_empty() => {
            machine_arm64 || !types.iter().all(|&t| t == CPU_TYPE_ARM64)
        }
        _ => true,
    }
}

/// The CPU types declared by a Mach-O header (thin or fat). `None` when the
/// bytes aren't recognizably Mach-O.
#[cfg(any(test, target_os = "macos"))]
fn macho_cpu_types(bytes: &[u8]) -> Option<Vec<u32>> {
    let word = |off: usize, be: bool| -> Option<u32> {
        let b: [u8; 4] = bytes.get(off..off + 4)?.try_into().ok()?;
        Some(if be {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    };
    const MH_MAGIC: u32 = 0xfeed_face;
    const MH_MAGIC_64: u32 = 0xfeed_facf;
    const FAT_MAGIC: u32 = 0xcafe_babe;
    const FAT_MAGIC_64: u32 = 0xcafe_babf;
    let be = word(0, true)?;
    match be {
        FAT_MAGIC | FAT_MAGIC_64 => {
            // Fat headers are big-endian; entries are 20 (`fat_arch`) or 32
            // (`fat_arch_64`) bytes. A Java class file shares FAT_MAGIC, so
            // bound the count to something a real binary would have.
            let entry_size = if be == FAT_MAGIC_64 { 32 } else { 20 };
            let count = word(4, true)? as usize;
            if count == 0 || count > 32 {
                return None;
            }
            (0..count).map(|i| word(8 + i * entry_size, true)).collect()
        }
        MH_MAGIC | MH_MAGIC_64 => Some(vec![word(4, true)?]),
        _ => {
            let le = word(0, false)?;
            if le == MH_MAGIC || le == MH_MAGIC_64 {
                Some(vec![word(4, false)?])
            } else {
                None
            }
        }
    }
}

/// Resolve a **`llamacpp`** backend's engine: the row's explicit
/// `engine_path` if set (a pin — set-but-missing resolves to `None`), else
/// discovery across `discovery_dirs` (typically `$PATH` then the usual
/// install prefixes). Pure over its injected inputs so it's unit-testable.
fn resolve_external_engine_path(
    engine_path: Option<&str>,
    discovery_dirs: impl Iterator<Item = PathBuf>,
    exists: &dyn Fn(&Path) -> bool,
) -> Option<PathBuf> {
    if let Some(p) = engine_path {
        let p = PathBuf::from(p);
        return exists(&p).then_some(p);
    }
    for dir in discovery_dirs {
        let candidate = dir.join("llama-server");
        if exists(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// The discovery search path for a user-owned `llama-server`: `$PATH` first,
/// then the usual install prefixes (a GUI launched from Finder inherits no
/// shell `$PATH`, so Homebrew's prefix is checked explicitly).
fn external_discovery_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default();
    for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
        dirs.push(PathBuf::from(dir));
    }
    dirs
}

/// A `llamacpp` backend's engine binary, resolved against the live system.
pub(crate) fn resolve_external_engine(engine_path: Option<&str>) -> Option<PathBuf> {
    resolve_external_engine_path(
        engine_path,
        external_discovery_dirs().into_iter(),
        &usable_engine_binary,
    )
}

/// A plain (non-attesting) HTTPS-capable client: native trust roots, used
/// for model downloads and loopback engine traffic.
pub(crate) fn plain_http_client() -> Result<reqwest::Client, AppError> {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(crate::load_native_root_store())
        .with_no_client_auth();
    reqwest::Client::builder()
        .tls_backend_preconfigured(tls_config)
        .user_agent(concat!("eidola-app-core/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| AppError::LocalModel {
            message: format!("constructing HTTP client: {e}"),
        })
}

/// Pick a free loopback port by binding port 0 and reading the assignment
/// back. The tiny bind→spawn race window is acceptable on loopback.
fn pick_free_port() -> Result<u16, AppError> {
    let listener =
        std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|e| AppError::LocalModel {
            message: format!("no free loopback port: {e}"),
        })?;
    let port = listener
        .local_addr()
        .map_err(|e| AppError::LocalModel {
            message: format!("no free loopback port: {e}"),
        })?
        .port();
    Ok(port)
}

fn sidecar_path(gguf_path: &Path) -> PathBuf {
    let mut os = gguf_path.as_os_str().to_owned();
    os.push(".meta.json");
    PathBuf::from(os)
}

// ============================================================================
// Inner methods — the operations `AppCore` wraps.
// ============================================================================

impl Inner {
    /// Snapshot the whole local-inference domain: resolved engine binary,
    /// the managed `local` store (on disk, downloading, loading, loaded),
    /// and every configured `llamacpp` backend's scanned directory.
    pub(crate) async fn local_models_state(&self) -> Result<LocalModelsState, AppError> {
        let cfg = self.load_config();
        let engine_path = resolve_local_engine(&cfg).map(|p| p.display().to_string());
        let dir = models_dir(&self.data_dir);

        let mut models: Vec<LocalModelInfo> = self
            .scan_engine_dir(crate::backends::LOCAL_BACKEND_ID, &dir)
            .await;

        // In-flight downloads (not yet on disk as `.gguf`) — the managed
        // local store only.
        {
            let downloads = self.local.downloads.lock().expect("downloads lock");
            for (slug, dl) in downloads.iter() {
                if models.iter().any(|m| &m.slug == slug) {
                    continue;
                }
                let total = dl.total.load(Ordering::Relaxed);
                models.push(LocalModelInfo {
                    id: engine_model_id(crate::backends::LOCAL_BACKEND_ID, slug),
                    slug: slug.clone(),
                    display_name: dl.display_name.clone(),
                    file_name: format!("{slug}.gguf"),
                    size_bytes: (total > 0).then_some(total),
                    source_url: Some(dl.source_url.clone()),
                    status: LocalModelStatus::Downloading {
                        received: dl.received.load(Ordering::Relaxed),
                        total: (total > 0).then_some(total),
                    },
                    last_error: None,
                });
            }
        }

        // Failures for local slugs with no file and no active download (a
        // failed download leaves nothing on disk — the error must stay
        // visible).
        {
            let failures = self.local.failures.lock().expect("failures lock");
            for ((backend_id, slug), message) in failures.iter() {
                if backend_id != crate::backends::LOCAL_BACKEND_ID
                    || models.iter().any(|m| &m.slug == slug)
                {
                    continue;
                }
                models.push(LocalModelInfo {
                    id: engine_model_id(crate::backends::LOCAL_BACKEND_ID, slug),
                    slug: slug.clone(),
                    display_name: prettify_stem(slug),
                    file_name: format!("{slug}.gguf"),
                    size_bytes: None,
                    source_url: None,
                    status: LocalModelStatus::Available,
                    last_error: Some(message.clone()),
                });
            }
        }

        models.sort_by(|a, b| {
            a.display_name
                .to_lowercase()
                .cmp(&b.display_name.to_lowercase())
        });

        // Configured llamacpp backends: scan each user-owned directory.
        // Disabled backends are still listed (Settings shows them greyed);
        // removed ones are not.
        let mut external = Vec::new();
        let db_conn = self.db_conn().await?;
        for row in crate::db::list_backends(&db_conn).await? {
            if row.kind != crate::backends::BackendKind::LlamaCpp.as_str() {
                continue;
            }
            let Some(models_dir) = row.models_dir.clone() else {
                continue;
            };
            let mut scanned = self.scan_engine_dir(&row.id, Path::new(&models_dir)).await;
            scanned.sort_by(|a, b| {
                a.display_name
                    .to_lowercase()
                    .cmp(&b.display_name.to_lowercase())
            });
            external.push(ExternalEngineBackend {
                backend_id: row.id,
                display_name: row.display_name,
                enabled: row.enabled,
                models_dir,
                engine_path: resolve_external_engine(row.engine_path.as_deref())
                    .map(|p| p.display().to_string()),
                auto_start: row.auto_start,
                models: scanned,
            });
        }

        Ok(LocalModelsState {
            engine_path,
            models,
            external,
        })
    }

    /// Scan one directory of `.gguf`s for a backend, merging live engine
    /// status and standing failures. Shared by the managed local store and
    /// the user-owned llamacpp directories.
    async fn scan_engine_dir(&self, backend_id: &str, dir: &Path) -> Vec<LocalModelInfo> {
        let mut models: Vec<LocalModelInfo> = Vec::new();
        if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if !name.to_ascii_lowercase().ends_with(".gguf") {
                    continue;
                }
                let slug = slug_for_file(name);
                let size_bytes = entry.metadata().await.ok().map(|m| m.len());
                let sidecar: Option<ModelSidecar> = tokio::fs::read(sidecar_path(&path))
                    .await
                    .ok()
                    .and_then(|bytes| serde_json::from_slice(&bytes).ok());

                let key = (backend_id.to_string(), slug.clone());
                let status = {
                    let engines = self.local.engines.lock().expect("engines lock");
                    match engines.get(&key) {
                        Some(e) if e.ready => LocalModelStatus::Loaded {
                            port: e.port,
                            context_tokens: e.context_tokens,
                            pinned: e.pinned,
                        },
                        Some(_) => LocalModelStatus::Loading,
                        None => LocalModelStatus::Available,
                    }
                };
                let last_error = self
                    .local
                    .failures
                    .lock()
                    .expect("failures lock")
                    .get(&key)
                    .cloned();

                let display_name = sidecar
                    .as_ref()
                    .map(|s| s.display_name.clone())
                    .unwrap_or_else(|| prettify_stem(&slug));

                models.push(LocalModelInfo {
                    id: engine_model_id(backend_id, &slug),
                    slug,
                    display_name,
                    file_name: name.to_string(),
                    size_bytes,
                    source_url: sidecar.map(|s| s.source_url),
                    status,
                    last_error,
                });
            }
        }
        models
    }

    /// Start downloading a model from `url` in the background. Returns the
    /// `<slug>@local` id immediately; progress and completion arrive as
    /// [`Change::LocalModels`] emissions. If the URL matches a curated
    /// catalog entry its display name is adopted.
    pub(crate) async fn download_local_model(&self, url: &str) -> Result<String, AppError> {
        let (download_url, file_name) = normalize_model_url(url)?;
        let slug = slug_for_file(&file_name);
        let id = engine_model_id(crate::backends::LOCAL_BACKEND_ID, &slug);
        let dir = models_dir(&self.data_dir);

        if dir.join(&file_name).exists() {
            return Err(AppError::LocalModel {
                message: format!("`{file_name}` is already downloaded"),
            });
        }

        let display_name = LOCAL_MODEL_CATALOG
            .iter()
            .find(|c| c.file_name == file_name || c.url == download_url)
            .map(|c| c.display_name.to_string())
            .unwrap_or_else(|| prettify_stem(&slug));

        let entry = Arc::new(DownloadEntry {
            display_name,
            source_url: download_url.clone(),
            received: AtomicU64::new(0),
            total: AtomicU64::new(0),
            cancel: AtomicBool::new(false),
        });
        {
            let mut downloads = self.local.downloads.lock().expect("downloads lock");
            if downloads.contains_key(&slug) {
                return Err(AppError::LocalModel {
                    message: format!("`{file_name}` is already downloading"),
                });
            }
            downloads.insert(slug.clone(), entry.clone());
        }
        let key = (crate::backends::LOCAL_BACKEND_ID.to_string(), slug.clone());
        self.local
            .failures
            .lock()
            .expect("failures lock")
            .remove(&key);
        self.bus.emit(Change::LocalModels);

        let client = match &self.http_override {
            Some(c) => c.clone(),
            None => plain_http_client()?,
        };
        let bus = self.bus.clone();
        let local = self.local.clone();
        let slug_task = slug.clone();
        // Core-owned transfer task: survives any window; cancellation is the
        // explicit flag, checked between chunks.
        tokio::spawn(async move {
            let result = run_download(&client, &download_url, &dir, &file_name, &entry, &bus).await;
            local
                .downloads
                .lock()
                .expect("downloads lock")
                .remove(&slug_task);
            if let Err(e) = result {
                local.failures.lock().expect("failures lock").insert(
                    (crate::backends::LOCAL_BACKEND_ID.to_string(), slug_task),
                    e.to_string(),
                );
            }
            bus.emit(Change::LocalModels);
        });

        Ok(id)
    }

    /// Cancel an in-flight download. The transfer task removes the partial
    /// file and emits when it notices (within one chunk).
    pub(crate) async fn cancel_local_model_download(&self, id: &str) -> Result<(), AppError> {
        let (_, slug) = engine_key_for_id(id);
        let downloads = self.local.downloads.lock().expect("downloads lock");
        match downloads.get(&slug) {
            Some(entry) => {
                entry.cancel.store(true, Ordering::Relaxed);
                Ok(())
            }
            None => Err(AppError::LocalModel {
                message: format!("no download in progress for `{slug}`"),
            }),
        }
    }

    /// Delete a downloaded model (its `.gguf` + sidecar). Refuses while the
    /// model is loaded or loading — unload first, explicitly. Only the
    /// managed `local` store deletes; llamacpp backends' files are the
    /// user's own.
    pub(crate) async fn delete_local_model(&self, id: &str) -> Result<(), AppError> {
        let key = engine_key_for_id(id);
        if key.0 != crate::backends::LOCAL_BACKEND_ID {
            return Err(AppError::LocalModel {
                message: format!(
                    "models in the `{}` backend's directory are managed by you, not Eidola — \
                     delete the file yourself if you mean to",
                    key.0
                ),
            });
        }
        let slug = key.1.clone();
        {
            let engines = self.local.engines.lock().expect("engines lock");
            if engines.contains_key(&key) {
                return Err(AppError::LocalModel {
                    message: format!("`{slug}` is loaded — unload it before deleting"),
                });
            }
        }
        {
            let downloads = self.local.downloads.lock().expect("downloads lock");
            if downloads.contains_key(&slug) {
                return Err(AppError::LocalModel {
                    message: format!("`{slug}` is downloading — cancel the download instead"),
                });
            }
        }
        if let Some(path) = find_model_file(&models_dir(&self.data_dir), &slug).await {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| AppError::LocalModel {
                    message: format!("failed to delete `{slug}`: {e}"),
                })?;
            let _ = tokio::fs::remove_file(sidecar_path(&path)).await;
        }
        // Also clears a lingering failure row when no file existed.
        self.local
            .failures
            .lock()
            .expect("failures lock")
            .remove(&key);
        self.bus.emit(Change::LocalModels);
        Ok(())
    }

    /// Load a model: spawn `llama-server` on a free loopback port and wait
    /// until its `/health` endpoint reports ready (or the load fails).
    /// Accepts `<slug>@<backend>` or a bare slug (shorthand for the
    /// managed local store).
    ///
    /// **Idempotent**: a ready engine returns immediately; one still
    /// warming up is awaited rather than double-spawned (so a request that
    /// races another request's auto-load simply joins it). Before spawning,
    /// the eviction planner ([`plan_evictions`]) makes room inside the
    /// memory budget by unloading least-recently-used engines that are
    /// neither pinned nor serving an in-flight turn; if even that can't
    /// free enough, the load is refused with an error naming what holds
    /// the memory (and nothing is unloaded).
    ///
    /// Emits [`Change::LocalModels`] at spawn, on ready, and on failure; a
    /// supervisor task owns the child and also emits if the engine later
    /// exits unexpectedly.
    pub(crate) async fn load_local_model(&self, id: &str) -> Result<(), AppError> {
        let key = engine_key_for_id(id);
        let (backend_id, slug) = (key.0.clone(), key.1.clone());
        let cfg = self.load_config();

        // Already present: ready → done; warming → join the in-flight load.
        if self.local.engine_present(&key) {
            if self.local.ready_engine(&backend_id, &slug).is_some() {
                return Ok(());
            }
            return self.await_engine_ready(&key).await;
        }

        // Resolve the model file's directory by backend, and note how the
        // engine binary resolves for it: the managed `local` store served by
        // the bundled engine, or a `llamacpp` backend's user-owned directory
        // served by the user's own (or discovered) `llama-server`. The engine
        // itself is resolved *after* the model-file check so a missing file
        // reports the clearer error.
        enum EngineSource {
            /// The bundled `local` engine.
            Bundled,
            /// A `llamacpp` backend's explicit path (`Some`) or discovery.
            External(Option<String>),
        }
        let (dir, engine_source) = if backend_id == crate::backends::LOCAL_BACKEND_ID {
            (models_dir(&self.data_dir), EngineSource::Bundled)
        } else {
            let db_conn = self.db_conn().await?;
            let row = self.require_backend(&db_conn, &backend_id).await?;
            if row.kind != crate::backends::BackendKind::LlamaCpp.as_str() {
                return Err(AppError::LocalModel {
                    message: format!("backend `{backend_id}` does not serve local engines"),
                });
            }
            let dir = PathBuf::from(row.models_dir.ok_or_else(|| AppError::LocalModel {
                message: format!("backend `{backend_id}` has no models directory"),
            })?);
            (dir, EngineSource::External(row.engine_path))
        };

        let model_path =
            find_model_file(&dir, &slug)
                .await
                .ok_or_else(|| AppError::LocalModel {
                    message: format!("no model file for `{slug}` in `{}`", dir.display()),
                })?;

        let engine = match engine_source {
            EngineSource::Bundled => {
                resolve_local_engine(&cfg).ok_or_else(|| AppError::LocalModel {
                    message: "this build doesn't include the bundled inference engine — set \
                              `llama_server_path` in config to point at a `llama-server` binary"
                        .into(),
                })?
            }
            EngineSource::External(engine_path) => resolve_external_engine(engine_path.as_deref())
                .ok_or_else(|| AppError::LocalModel {
                    message: format!(
                        "llama-server not found for backend `{backend_id}` — install \
                             llama.cpp (e.g. `brew install llama.cpp`) or set its engine path"
                    ),
                })?,
        };

        let port = pick_free_port()?;
        // The alias equals the selectable model id, so a chat body's
        // `model` field matches what the engine expects verbatim.
        let model_id = engine_model_id(&backend_id, &slug);

        let footprint = engine_footprint(
            tokio::fs::metadata(&model_path)
                .await
                .map(|m| m.len())
                .unwrap_or(0),
        );

        // Make room: plan LRU evictions and insert this load's warming
        // entry in one critical section ([`LocalRuntime::reserve_engine`]),
        // so concurrent loads of different models can't double-book the
        // budget. A refusal reserves and unloads nothing.
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let reserved = self
            .local
            .reserve_engine(&key, port, footprint, shutdown_tx)
            .map_err(|message| AppError::LocalModel {
                message: format!("cannot load `{model_id}`: {message}"),
            })?;
        let Some(victims) = reserved else {
            // Raced another load between the presence check and here — join
            // that load instead of double-spawning.
            return self.await_engine_ready(&key).await;
        };
        // Until the supervisor takes ownership at spawn, an error return
        // (or this future being dropped) must roll the reservation back.
        let reservation = ReservationGuard {
            local: self.local.clone(),
            bus: self.bus.clone(),
            key: Some(key.clone()),
        };
        self.local
            .failures
            .lock()
            .expect("failures lock")
            .remove(&key);
        self.bus.emit(Change::LocalModels);

        for victim in victims {
            // Best-effort: a victim may have been unloaded concurrently.
            let _ = self
                .unload_local_model(&engine_model_id(&victim.0, &victim.1))
                .await;
        }

        let mut command = tokio::process::Command::new(&engine);
        command
            .arg("-m")
            .arg(&model_path)
            .args(["--host", "127.0.0.1"])
            .args(["--port", &port.to_string()])
            .args(["--alias", &model_id])
            .args(["-c", &LOCAL_CONTEXT_TOKENS.to_string()])
            .args(["-ngl", "999"])
            .arg("--jinja")
            .arg("--no-webui")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
        let bus = self.bus.clone();
        let local = self.local.clone();
        let http = match &self.http_override {
            Some(c) => c.clone(),
            None => plain_http_client()?,
        };
        reservation.defuse();
        // Supervisor task: owns the child for its whole life. Cancellation
        // authority is the shutdown channel (map removal → send) — and if
        // the whole runtime is torn down, `kill_on_drop` reaps the child.
        tokio::spawn(async move {
            supervise_engine(command, port, http, shutdown_rx, ready_tx, bus, local, key).await;
        });

        match ready_rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(message)) => Err(AppError::LocalModel { message }),
            Err(_) => Err(AppError::LocalModel {
                message: "engine supervisor exited before the model became ready".into(),
            }),
        }
    }

    /// Wait for another caller's in-flight load of `key` to settle: ready →
    /// `Ok`, entry gone (load failed / unloaded) → the recorded failure, or
    /// a timeout mirroring the loader's own budget.
    async fn await_engine_ready(&self, key: &EngineKey) -> Result<(), AppError> {
        let deadline = std::time::Instant::now() + ENGINE_READY_TIMEOUT;
        loop {
            if self.local.ready_engine(&key.0, &key.1).is_some() {
                return Ok(());
            }
            if !self.local.engine_present(key) {
                let message = self
                    .local
                    .failures
                    .lock()
                    .expect("failures lock")
                    .get(key)
                    .cloned()
                    .unwrap_or_else(|| "the engine load was cancelled".into());
                return Err(AppError::LocalModel { message });
            }
            if std::time::Instant::now() >= deadline {
                return Err(AppError::LocalModel {
                    message: format!(
                        "engine did not become ready within {}s",
                        ENGINE_READY_TIMEOUT.as_secs()
                    ),
                });
            }
            tokio::time::sleep(ENGINE_POLL_INTERVAL).await;
        }
    }

    /// Pin or unpin a loaded engine. Pinned engines are protected from
    /// automatic (LRU) unloading; manual unload still applies. Runtime
    /// state — it lives and dies with the loaded engine.
    pub(crate) async fn set_local_model_pinned(
        &self,
        id: &str,
        pinned: bool,
    ) -> Result<(), AppError> {
        let key = engine_key_for_id(id);
        {
            let mut engines = self.local.engines.lock().expect("engines lock");
            let entry = engines.get_mut(&key).ok_or_else(|| AppError::LocalModel {
                message: format!(
                    "`{}` is not loaded — only loaded models can be pinned",
                    engine_model_id(&key.0, &key.1)
                ),
            })?;
            entry.pinned = pinned;
        }
        self.bus.emit(Change::LocalModels);
        Ok(())
    }

    /// Unload a model: signal its supervisor, which kills the subprocess.
    pub(crate) async fn unload_local_model(&self, id: &str) -> Result<(), AppError> {
        let key = engine_key_for_id(id);
        let entry = {
            let mut engines = self.local.engines.lock().expect("engines lock");
            engines.remove(&key)
        };
        match entry {
            Some(e) => {
                // Supervisor may already be gone (crash path); either way the
                // map entry is removed, which is the user-visible state.
                let _ = e.shutdown.send(());
                self.bus.emit(Change::LocalModels);
                Ok(())
            }
            None => Err(AppError::LocalModel {
                message: format!("`{}` is not loaded", engine_model_id(&key.0, &key.1)),
            }),
        }
    }

    /// Signal *every* live engine supervisor to kill its subprocess, and
    /// return how many were signalled. Idempotent: the map is drained, so a
    /// second call returns 0.
    ///
    /// This reads the **engine registry** and nothing else — no filesystem,
    /// no database, no `Result`. That is the whole point of its existing
    /// separately from a loop over [`Self::local_models_state`]: that
    /// snapshot is reconstructed by *scanning* the managed models directory
    /// and every `llamacpp` backend's directory (one `read_dir`, a `stat`
    /// and a sidecar read per `.gguf`, plus a DB round trip to list the
    /// backends), consulting this map only to *decorate* a file it already
    /// found. A running engine whose `.gguf` was renamed or deleted
    /// mid-session is therefore absent from that snapshot while its
    /// subprocess is very much alive — and a slow or large directory would
    /// spend a shutdown budget on I/O before killing anything. Neither is
    /// acceptable on a quit path.
    ///
    /// Draining alone would not be enough: a load already past its `await`s
    /// but not yet at [`LocalRuntime::reserve_engine`] would resume after
    /// the drain and spawn a subprocess into a process that is about to
    /// `exit()`. So this also sets the one-way shutdown latch — **inside
    /// the same lock the reservation takes**, so there is no instant in
    /// which the registry is empty and loads are still permitted.
    ///
    /// Like [`Self::unload_local_model`], this *signals*; the supervisor
    /// task owns the child and does the `start_kill`. Callers on a quit
    /// path must leave the runtime a moment to run them.
    ///
    /// **It deliberately does not emit `Change::LocalModels`.** Every other
    /// engine transition does, because something is watching; here nothing
    /// is — the process is exiting and no consumer can render the change.
    /// Emitting was actively harmful: the GUI's app-lifetime bus bridge is a
    /// foreground task that gpui keeps driving through its bounded shutdown
    /// block, *after* `App::shutdown` has set `quitting`, so the dispatch
    /// would reach `LocalModelsStore::refresh` → `cx.spawn` → gpui's
    /// "Can't spawn on main thread after on_app_quit" panic. The one caller
    /// is the quit hook (`AppCore::shutdown_engines`); there is no other,
    /// and a new one would want this same silence.
    pub(crate) fn shutdown_all_engines(&self) -> usize {
        let entries: Vec<EngineEntry> = {
            let mut engines = self.local.engines.lock().expect("engines lock");
            self.local.shutting_down.store(true, Ordering::SeqCst);
            engines.drain().map(|(_, entry)| entry).collect()
        };
        let count = entries.len();
        for entry in entries {
            // Supervisor may already be gone (crash path); the map removal
            // is the user-visible state either way.
            let _ = entry.shutdown.send(());
        }
        count
    }
}

// ============================================================================
// Download transfer
// ============================================================================

async fn remove_partial(path: &Path) {
    let _ = tokio::fs::remove_file(path).await;
}

/// Stream `url` to `<dir>/<file_name>.part`, then write the sidecar and
/// rename into place. Cancellation is checked between chunks; any failure
/// removes the partial file. Returns `Ok` for both completion and
/// cancellation — only real failures are recorded.
async fn run_download(
    client: &reqwest::Client,
    url: &str,
    dir: &Path,
    file_name: &str,
    entry: &DownloadEntry,
    bus: &BroadcastSource,
) -> Result<(), AppError> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    tokio::fs::create_dir_all(dir)
        .await
        .map_err(|e| AppError::LocalModel {
            message: format!("failed to create models directory: {e}"),
        })?;
    let part_path = dir.join(format!("{file_name}.part"));
    let final_path = dir.join(file_name);

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| AppError::LocalModel {
            message: format!("download failed: {e}"),
        })?;
    if !response.status().is_success() {
        return Err(AppError::LocalModel {
            message: format!("download failed: HTTP {}", response.status().as_u16()),
        });
    }
    if let Some(len) = response.content_length() {
        entry.total.store(len, Ordering::Relaxed);
    }

    let mut file = tokio::fs::File::create(&part_path)
        .await
        .map_err(|e| AppError::LocalModel {
            message: format!("failed to create file: {e}"),
        })?;

    let mut stream = response.bytes_stream();
    let mut last_emit = std::time::Instant::now();
    while let Some(chunk) = stream.next().await {
        if entry.cancel.load(Ordering::Relaxed) {
            drop(file);
            remove_partial(&part_path).await;
            return Ok(());
        }
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                drop(file);
                remove_partial(&part_path).await;
                return Err(AppError::LocalModel {
                    message: format!("download interrupted: {e}"),
                });
            }
        };
        if let Err(e) = file.write_all(&bytes).await {
            drop(file);
            remove_partial(&part_path).await;
            return Err(AppError::LocalModel {
                message: format!("failed to write file: {e}"),
            });
        }
        entry
            .received
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        if last_emit.elapsed() >= PROGRESS_EMIT_INTERVAL {
            last_emit = std::time::Instant::now();
            bus.emit(Change::LocalModels);
        }
    }

    if let Err(e) = file.sync_all().await {
        drop(file);
        remove_partial(&part_path).await;
        return Err(AppError::LocalModel {
            message: format!("failed to flush file: {e}"),
        });
    }
    drop(file);

    let sidecar = ModelSidecar {
        display_name: entry.display_name.clone(),
        source_url: entry.source_url.clone(),
        downloaded_at_ms: crate::now_ms(),
    };
    let sidecar_bytes = serde_json::to_vec_pretty(&sidecar).unwrap_or_default();
    let _ = tokio::fs::write(sidecar_path(&final_path), sidecar_bytes).await;

    tokio::fs::rename(&part_path, &final_path)
        .await
        .map_err(|e| AppError::LocalModel {
            message: format!("failed to finalize download: {e}"),
        })?;
    Ok(())
}

// ============================================================================
// Engine supervision
// ============================================================================

/// Own a `llama-server` child for its whole life: spawn, poll `/health`
/// until ready, then wait for shutdown or unexpected exit. Every state
/// transition updates the engines map and emits [`Change::LocalModels`].
#[allow(clippy::too_many_arguments)]
async fn supervise_engine(
    mut command: tokio::process::Command,
    port: u16,
    http: reqwest::Client,
    mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    ready_tx: tokio::sync::oneshot::Sender<Result<(), String>>,
    bus: BroadcastSource,
    local: Arc<LocalRuntime>,
    key: EngineKey,
) {
    let fail = |message: &str| {
        local.engines.lock().expect("engines lock").remove(&key);
        local
            .failures
            .lock()
            .expect("failures lock")
            .insert(key.clone(), message.to_string());
    };

    // **Every** emission in this supervisor goes through here, and it is
    // silent once the quit-time shutdown latch is set.
    //
    // Silencing only [`Inner::shutdown_all_engines`] left this second emitter
    // wide open: the drain merely *signals*, and the supervisor's own
    // shutdown arms emit right after killing the child — into the same window
    // that panics. The GUI's bus bridge is an app-lifetime foreground task
    // gpui keeps driving through its bounded shutdown block *after*
    // `App::shutdown` sets `quitting`, so any dispatch there reaches
    // `LocalModelsStore::refresh` → `cx.spawn` → "Can't spawn on main thread
    // after on_app_quit".
    //
    // Gating on the latch rather than on the shutdown *arm* is deliberate:
    // during a quit no path should emit, including a child that happens to
    // exit on its own at that moment. An ordinary unload is not a quit — the
    // latch is unset — so it still emits, which is what the Local settings
    // pane redraws from.
    let emit = || {
        if !local.shutting_down.load(Ordering::SeqCst) {
            bus.emit(Change::LocalModels);
        }
    };

    // A quit may have landed between our reservation and this task's first
    // poll; starting a subprocess now would orphan it to the imminent
    // `exit()`. Checked under the drain's own lock — see
    // `LocalRuntime::spawn_unless_shutting_down`.
    let Some(spawned) = local.spawn_unless_shutting_down(|| command.spawn()) else {
        let _ = ready_tx.send(Err("Eidola is shutting down".into()));
        return;
    };
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => {
            let message = format!("failed to start llama-server: {e}");
            fail(&message);
            emit();
            let _ = ready_tx.send(Err(message));
            return;
        }
    };

    // Drain stderr into a bounded tail so a failed load carries the
    // engine's actual complaint instead of a generic timeout.
    let stderr_tail: Arc<StdMutex<std::collections::VecDeque<String>>> =
        Arc::new(StdMutex::new(std::collections::VecDeque::new()));
    if let Some(stderr) = child.stderr.take() {
        let tail = stderr_tail.clone();
        tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let mut lines = tokio::io::BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut tail = tail.lock().expect("stderr tail lock");
                if tail.len() >= 30 {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
        });
    }
    let tail_text = || {
        let tail = stderr_tail.lock().expect("stderr tail lock");
        let lines: Vec<String> = tail.iter().cloned().collect();
        let start = lines.len().saturating_sub(6);
        lines[start..].join("\n")
    };

    let health_url = format!("http://127.0.0.1:{port}/health");
    let deadline = std::time::Instant::now() + ENGINE_READY_TIMEOUT;

    // Phase 1: wait for readiness. The child's exit is observed via
    // `try_wait` between polls so no `&mut child` borrow spans the select.
    enum LoadEnd {
        Ready,
        Shutdown,
        Exited(Option<i32>),
        TimedOut,
    }
    let end = loop {
        let slept = tokio::select! {
            _ = &mut shutdown_rx => false,
            _ = tokio::time::sleep(ENGINE_POLL_INTERVAL) => true,
        };
        if !slept {
            break LoadEnd::Shutdown;
        }
        if let Ok(Some(status)) = child.try_wait() {
            break LoadEnd::Exited(status.code());
        }
        if std::time::Instant::now() >= deadline {
            break LoadEnd::TimedOut;
        }
        // The probe must race the shutdown signal, not precede it.
        //
        // Awaiting the request outside the `select!` meant the oneshot was
        // simply not polled for its duration — and a *warming* engine is
        // exactly the case where `/health` can hang rather than refuse: the
        // socket is accepted while a multi-gigabyte model loads, and this
        // client has no request timeout. A quit landing in that window sent
        // its signal into a receiver nobody was watching, the drain's brief
        // grace expired, `exit()` followed, and the child outlived the
        // process — the orphan the whole teardown exists to prevent.
        let ready = tokio::select! {
            _ = &mut shutdown_rx => break LoadEnd::Shutdown,
            resp = http.get(&health_url).send() => {
                matches!(resp, Ok(r) if r.status().is_success())
            }
        };
        if ready {
            break LoadEnd::Ready;
        }
    };

    match end {
        LoadEnd::Ready => {}
        LoadEnd::Shutdown => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            // Map entry was already removed by `unload`.
            emit();
            let _ = ready_tx.send(Err("load cancelled".into()));
            return;
        }
        LoadEnd::Exited(code) => {
            let message = format!(
                "llama-server exited during load (status {code:?}). Last output:\n{}",
                tail_text(),
            );
            fail(&message);
            emit();
            let _ = ready_tx.send(Err(message));
            return;
        }
        LoadEnd::TimedOut => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            let message = format!(
                "llama-server did not become ready within {}s. Last output:\n{}",
                ENGINE_READY_TIMEOUT.as_secs(),
                tail_text(),
            );
            fail(&message);
            emit();
            let _ = ready_tx.send(Err(message));
            return;
        }
    }

    // Ready — flip the map entry (it may have been removed by a concurrent
    // unload, in which case we shut down instead of serving a ghost). The
    // decision happens under the lock; the kill happens after it drops.
    let still_wanted = {
        let mut engines = local.engines.lock().expect("engines lock");
        match engines.get_mut(&key) {
            Some(e) => {
                e.ready = true;
                true
            }
            None => false,
        }
    };
    if !still_wanted {
        let _ = child.start_kill();
        let _ = child.wait().await;
        let _ = ready_tx.send(Err("unloaded during startup".into()));
        return;
    }
    emit();
    let _ = ready_tx.send(Ok(()));

    // Phase 2: serve until shutdown or unexpected exit. The `child.wait()`
    // borrow ends with the select expression, freeing `child` for the kill.
    let exited = tokio::select! {
        _ = &mut shutdown_rx => None,
        exit = child.wait() => Some(exit),
    };
    match exited {
        None => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            emit();
        }
        Some(exit) => {
            let code = exit.ok().and_then(|s| s.code());
            let message = format!(
                "llama-server exited unexpectedly (status {code:?}). Last output:\n{}",
                tail_text(),
            );
            fail(&message);
            emit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_accepts_direct_gguf_urls() {
        let (url, name) = normalize_model_url(
            "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf",
        )
        .unwrap();
        assert!(url.ends_with("gemma-4-E2B_q4_0-it.gguf"));
        assert_eq!(name, "gemma-4-E2B_q4_0-it.gguf");
    }

    #[test]
    fn normalize_rewrites_hf_blob_urls_and_strips_query() {
        let (url, name) = normalize_model_url(
            "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/blob/main/gemma-4-12b-it-qat-q4_0.gguf?download=true",
        )
        .unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/gemma-4-12b-it-qat-q4_0.gguf"
        );
        assert_eq!(name, "gemma-4-12b-it-qat-q4_0.gguf");
    }

    #[test]
    fn normalize_rejects_non_gguf_and_non_http() {
        assert!(normalize_model_url("https://example.com/model.bin").is_err());
        assert!(normalize_model_url("ftp://example.com/model.gguf").is_err());
        assert!(normalize_model_url("").is_err());
        assert!(normalize_model_url("https://example.com/").is_err());
    }

    #[test]
    fn catalog_entries_are_consistent() {
        for entry in LOCAL_MODEL_CATALOG {
            let (url, name) = normalize_model_url(entry.url).expect("catalog URL must normalize");
            assert_eq!(url, entry.url, "catalog URLs must already be direct");
            assert_eq!(name, entry.file_name);
            assert!(entry.size_bytes > 1_000_000_000);
        }
    }

    #[test]
    fn slug_round_trips_through_id() {
        let slug = slug_for_file("gemma-4-E2B_q4_0-it.gguf");
        assert_eq!(slug, "gemma-4-E2B_q4_0-it");
        assert_eq!(engine_model_id("local", &slug), "gemma-4-E2B_q4_0-it@local");
    }

    #[test]
    fn slug_strips_any_case_variant_of_the_extension() {
        // The scan filter is case-insensitive, so the slug strip must be
        // too — otherwise a `model.GGUF` is advertised but can never be
        // loaded (its path would be reconstructed as `model.gguf`).
        assert_eq!(slug_for_file("model.GGUF"), "model");
        assert_eq!(slug_for_file("model.GgUf"), "model");
        assert_eq!(slug_for_file("model.gguf"), "model");
        assert_eq!(slug_for_file("model.bin"), "model.bin");
        assert_eq!(slug_for_file(".gguf"), "");
    }

    // -- The eviction planner -------------------------------------------

    const GIB: u64 = 1 << 30;

    fn usage(
        name: &str,
        footprint: u64,
        last_used: i64,
        pinned: bool,
        in_flight: u64,
    ) -> EngineUsage {
        EngineUsage {
            key: ("local".to_string(), name.to_string()),
            footprint,
            last_used_ms: last_used,
            pinned,
            in_flight,
            warming: false,
        }
    }

    #[test]
    fn plan_no_eviction_when_it_fits() {
        let loaded = vec![usage("a", 4 * GIB, 1, false, 0)];
        assert_eq!(
            plan_evictions(3 * GIB, 16 * GIB, &loaded).unwrap(),
            Vec::<EngineKey>::new()
        );
    }

    #[test]
    fn plan_evicts_lru_first_and_only_as_needed() {
        // Budget 16, loaded 4+5+4=13, need 6 → free 3; evicting the LRU
        // (b, oldest) frees 5 more → 8 ≥ 6. One eviction, the oldest.
        let loaded = vec![
            usage("a", 4 * GIB, 300, false, 0),
            usage("b", 5 * GIB, 100, false, 0),
            usage("c", 4 * GIB, 200, false, 0),
        ];
        let plan = plan_evictions(6 * GIB, 16 * GIB, &loaded).unwrap();
        assert_eq!(plan, vec![("local".to_string(), "b".to_string())]);

        // Needing more takes the next-oldest too (b then c), never a.
        let plan = plan_evictions(10 * GIB, 16 * GIB, &loaded).unwrap();
        assert_eq!(
            plan,
            vec![
                ("local".to_string(), "b".to_string()),
                ("local".to_string(), "c".to_string()),
            ]
        );
    }

    #[test]
    fn plan_never_touches_pinned_or_in_flight_engines() {
        let loaded = vec![
            usage("pinned-old", 6 * GIB, 1, true, 0),
            usage("busy-old", 6 * GIB, 2, false, 3),
            usage("idle", 3 * GIB, 900, false, 0),
        ];
        // Fits after evicting only the idle one — the older pinned/busy
        // engines are skipped despite being better LRU candidates.
        let plan = plan_evictions(4 * GIB, 16 * GIB, &loaded).unwrap();
        assert_eq!(plan, vec![("local".to_string(), "idle".to_string())]);

        // Can't fit even after the idle eviction: refuse, naming the hold.
        let err = plan_evictions(8 * GIB, 16 * GIB, &loaded).unwrap_err();
        assert!(err.contains("pinned or in-use"), "got {err}");
    }

    #[test]
    fn plan_refuses_models_larger_than_the_budget() {
        let err = plan_evictions(20 * GIB, 16 * GIB, &[]).unwrap_err();
        assert!(err.contains("memory budget"), "got {err}");
    }

    #[test]
    fn plan_never_evicts_a_warming_engine() {
        // A warming entry is another load's reservation: its memory is
        // committed but there is no engine to gracefully unload yet.
        let mut warming = usage("warming", 6 * GIB, 1, false, 0);
        warming.warming = true;
        let err = plan_evictions(6 * GIB, 10 * GIB, &[warming]).unwrap_err();
        assert!(err.contains("held"), "got {err}");
    }

    #[test]
    fn reserve_engine_is_atomic_across_models() {
        // Two loads that would each fit alone must not both fit: the first
        // reservation is visible to (and protected from) the second plan,
        // because planning and reserving share one critical section.
        let runtime = LocalRuntime::default();
        runtime.set_memory_budget_for_test(10 * GIB);

        let key_a = ("local".to_string(), "a".to_string());
        let (tx_a, _rx_a) = tokio::sync::oneshot::channel();
        assert_eq!(
            runtime.reserve_engine(&key_a, 4001, 6 * GIB, tx_a).unwrap(),
            Some(Vec::new()),
            "the first load fits with no evictions"
        );

        let key_b = ("local".to_string(), "b".to_string());
        let (tx_b, _rx_b) = tokio::sync::oneshot::channel();
        let err = runtime
            .reserve_engine(&key_b, 4002, 6 * GIB, tx_b)
            .unwrap_err();
        assert!(err.contains("held"), "got {err}");
        assert!(
            !runtime.engine_present(&key_b),
            "a refused reservation must leave nothing behind"
        );

        // Re-reserving an existing key joins rather than double-books.
        let (tx_a2, _rx_a2) = tokio::sync::oneshot::channel();
        assert_eq!(
            runtime
                .reserve_engine(&key_a, 4003, 6 * GIB, tx_a2)
                .unwrap(),
            None
        );
    }

    #[test]
    fn the_shutdown_latch_refuses_a_reservation_that_arrives_after_the_drain() {
        // A load that was already awaiting (backend lookup, port pick,
        // `fs::metadata`) when the quit landed resumes *after* the drain saw
        // an empty registry. Without the latch it would reserve and spawn a
        // subprocess into a process about to `exit()`. The latch is checked
        // inside the reservation's own critical section, under the same lock
        // the drain takes, so there is no window between the two.
        let runtime = LocalRuntime::default();
        runtime.set_memory_budget_for_test(10 * GIB);
        runtime.shutting_down.store(true, Ordering::SeqCst);

        let key = ("local".to_string(), "late".to_string());
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let err = runtime.reserve_engine(&key, 4004, GIB, tx).unwrap_err();
        assert!(err.contains("shutting down"), "got {err}");
        assert!(
            !runtime.engine_present(&key),
            "a refused reservation leaves nothing to spawn against"
        );
    }

    #[test]
    fn the_shutdown_latch_also_refuses_a_spawn_that_was_already_reserved() {
        // The latch on `reserve_engine` closes the window before a
        // reservation; this closes the one after it. A supervisor whose
        // reservation was accepted a moment before the quit lands can have
        // its first poll scheduled *after* the drain walked the registry —
        // and an unconditional `command.spawn()` there starts a subprocess
        // into a process about to `exit()`.
        let runtime = LocalRuntime::default();
        assert_eq!(
            runtime.spawn_unless_shutting_down(|| "spawned"),
            Some("spawned"),
            "an ordinary load spawns"
        );

        runtime.shutting_down.store(true, Ordering::SeqCst);
        assert_eq!(
            runtime.spawn_unless_shutting_down(|| "spawned"),
            None,
            "once the drain has run, nothing may start a child"
        );
    }

    // -- Mach-O architecture gate --------------------------------------

    #[test]
    fn macho_cpu_types_reads_thin_and_fat_headers() {
        const CPU_TYPE_X86_64: u32 = 0x0100_0007;
        // Thin arm64 (little-endian on disk, as produced on macOS).
        let mut thin = 0xfeed_facf_u32.to_le_bytes().to_vec();
        thin.extend(CPU_TYPE_ARM64.to_le_bytes());
        assert_eq!(macho_cpu_types(&thin), Some(vec![CPU_TYPE_ARM64]));

        // Fat with both slices (big-endian header, 20-byte entries).
        let mut fat = 0xcafe_babe_u32.to_be_bytes().to_vec();
        fat.extend(2_u32.to_be_bytes());
        for cpu in [CPU_TYPE_X86_64, CPU_TYPE_ARM64] {
            fat.extend(cpu.to_be_bytes());
            fat.extend([0u8; 16]); // cpusubtype, offset, size, align
        }
        assert_eq!(
            macho_cpu_types(&fat),
            Some(vec![CPU_TYPE_X86_64, CPU_TYPE_ARM64])
        );

        // Not Mach-O: an ELF header, or a Java class file (which shares
        // FAT_MAGIC but has an implausible entry count).
        assert_eq!(macho_cpu_types(b"\x7fELF\x02\x01\x01\x00"), None);
        let mut class = 0xcafe_babe_u32.to_be_bytes().to_vec();
        class.extend(65_u32.to_be_bytes()); // minor+major version words
        assert_eq!(macho_cpu_types(&class), None);
        assert_eq!(macho_cpu_types(b""), None);
    }

    #[test]
    fn macho_compatibility_rejects_only_arm64_only_on_intel() {
        const CPU_TYPE_X86_64: u32 = 0x0100_0007;
        // The shipped sidecar case: arm64-only binary on an Intel Mac.
        assert!(!macho_machine_compatible(Some(&[CPU_TYPE_ARM64]), false));
        // Same binary on Apple Silicon is fine.
        assert!(macho_machine_compatible(Some(&[CPU_TYPE_ARM64]), true));
        // A universal or x86_64 binary passes everywhere.
        assert!(macho_machine_compatible(
            Some(&[CPU_TYPE_X86_64, CPU_TYPE_ARM64]),
            false
        ));
        assert!(macho_machine_compatible(Some(&[CPU_TYPE_X86_64]), false));
        // Unknown formats pass — the spawn surfaces real errors.
        assert!(macho_machine_compatible(None, false));
        assert!(macho_machine_compatible(Some(&[]), false));
    }

    // -- Engine resolution --------------------------------------------

    #[test]
    fn local_engine_override_is_a_pin() {
        // Override present + exists → used; present + missing → None (no
        // fall-through to env/exe; the escape hatch is honest).
        let exe = PathBuf::from("/app/Contents/MacOS/Eidola");
        assert_eq!(
            resolve_local_engine_path(
                Some("/opt/llama-server"),
                Some("/env/llama-server"),
                Some(&exe),
                &|p| p == Path::new("/opt/llama-server")
            ),
            Some(PathBuf::from("/opt/llama-server"))
        );
        assert_eq!(
            resolve_local_engine_path(
                Some("/nope"),
                Some("/env/llama-server"),
                Some(&exe),
                &|_| { false }
            ),
            None
        );
    }

    #[test]
    fn local_engine_env_then_exe_relative() {
        let exe = PathBuf::from("/app/Contents/MacOS/Eidola");
        // Env used when it exists.
        assert_eq!(
            resolve_local_engine_path(None, Some("/env/llama-server"), Some(&exe), &|p| p
                == Path::new("/env/llama-server")),
            Some(PathBuf::from("/env/llama-server"))
        );
        // Env missing → macOS .app sidecar under Contents/Resources/bin.
        let sidecar = PathBuf::from("/app/Contents/Resources/bin/llama-server");
        assert_eq!(
            resolve_local_engine_path(None, Some("/env/missing"), Some(&exe), &|p| p == sidecar),
            Some(sidecar)
        );
        // CLI/dev layout: a sibling next to the exe.
        let cli_exe = PathBuf::from("/nix/store/x/bin/eidola");
        let sibling = PathBuf::from("/nix/store/x/bin/llama-server");
        assert_eq!(
            resolve_local_engine_path(None, None, Some(&cli_exe), &|p| p == sibling),
            Some(sibling)
        );
    }

    #[test]
    fn local_engine_never_scans_path() {
        // A llama-server on $PATH must NOT satisfy the bundled `local`
        // engine — only override/env/exe-relative do. With nothing set and
        // no exe-relative match, resolution is None even though the probe
        // would happily report a $PATH binary present.
        let exe = PathBuf::from("/app/Contents/MacOS/Eidola");
        assert_eq!(
            resolve_local_engine_path(None, None, Some(&exe), &|p| p
                == Path::new("/usr/local/bin/llama-server")),
            None
        );
    }

    #[test]
    fn external_engine_path_is_preferred_over_discovery() {
        let dirs = || vec![PathBuf::from("/usr/bin")].into_iter();
        // engine_path set + exists → used even though discovery would match.
        assert_eq!(
            resolve_external_engine_path(Some("/custom/llama-server"), dirs(), &|_| true),
            Some(PathBuf::from("/custom/llama-server"))
        );
        // engine_path set + missing → None (a pin, no fall-through).
        assert_eq!(
            resolve_external_engine_path(Some("/custom/llama-server"), dirs(), &|p| p
                == Path::new("/usr/bin/llama-server")),
            None
        );
        // No engine_path → discovery finds it under a search dir.
        assert_eq!(
            resolve_external_engine_path(None, dirs(), &|p| p
                == Path::new("/usr/bin/llama-server")),
            Some(PathBuf::from("/usr/bin/llama-server"))
        );
        // No engine_path, nothing on the search path → None.
        assert_eq!(resolve_external_engine_path(None, dirs(), &|_| false), None);
    }

    #[test]
    fn engine_ids_and_keys_round_trip_per_backend() {
        // Every engine backend uses the uniform qualified form; the alias
        // equals the selection id.
        assert_eq!(engine_model_id("local", "tiny"), "tiny@local");
        assert_eq!(
            engine_key_for_id("tiny@local"),
            ("local".to_string(), "tiny".to_string())
        );
        assert_eq!(engine_model_id("my-box", "tiny"), "tiny@my-box");
        assert_eq!(
            engine_key_for_id("tiny@my-box"),
            ("my-box".to_string(), "tiny".to_string())
        );
        // A bare slug is the managed-store shorthand for the management
        // verbs (`eidola model load tiny`).
        assert_eq!(
            engine_key_for_id("tiny"),
            ("local".to_string(), "tiny".to_string())
        );
    }
}
