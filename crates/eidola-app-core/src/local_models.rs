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
//! OpenAI-HTTP seam a future *external* llama.cpp instance or a self-hosted
//! second-device server would plug into.
//!
//! ## State
//!
//! Durable truth is the filesystem: `<data_dir>/models/` holds one
//! `<file>.gguf` per model plus a `<file>.gguf.meta.json` sidecar
//! (display name, source URL, download time). A `.gguf` dropped in
//! manually is picked up on the next scan — the sidecar is optional.
//! In-flight downloads write `<file>.gguf.part` and rename on completion,
//! so a crash never leaves a truncated file masquerading as a model.
//!
//! Runtime state (downloads in progress, running engines) lives in
//! [`LocalRuntime`] on `Inner`. Every state transition emits
//! [`Change::LocalModels`] so subscribers re-snapshot via
//! `AppCore::local_models_state`; download progress emits throttled.
//! Model identifiers are namespaced as `local/<file-stem>` so the chat
//! path can route on the prefix.

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

/// Namespace prefix that routes a model id to the local engine instead of
/// the remote attested server.
pub const LOCAL_MODEL_PREFIX: &str = "local/";

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
    /// Serving on `127.0.0.1:<port>`; selectable for chat.
    Loaded { port: u16, context_tokens: u32 },
}

/// One local model as shown in Settings → Models and (when loaded) the
/// model picker.
#[derive(Clone, Debug)]
pub struct LocalModelInfo {
    /// The chat-routable id: `local/<slug>`.
    pub id: String,
    /// The file stem — the part after `local/`.
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

/// Snapshot of the whole local-inference domain.
#[derive(Clone, Debug)]
pub struct LocalModelsState {
    /// Resolved `llama-server` binary (config override, `PATH`, or a known
    /// install location), if one was found.
    pub engine_path: Option<String>,
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
}

/// All runtime (non-durable) local-inference state, held by `Inner` behind
/// an `Arc` so transfer/supervisor tasks can outlive individual calls.
/// Plain `std::sync::Mutex` — never held across an `.await`.
#[derive(Default)]
pub(crate) struct LocalRuntime {
    downloads: StdMutex<HashMap<String, Arc<DownloadEntry>>>,
    engines: StdMutex<HashMap<String, EngineEntry>>,
    /// Last failure per slug (download or load), shown until retried.
    failures: StdMutex<HashMap<String, String>>,
}

impl LocalRuntime {
    /// The loopback base URL + context window of a ready engine, if the
    /// slug is loaded. `None` while still warming up.
    pub(crate) fn ready_engine(&self, slug: &str) -> Option<(String, u32)> {
        let engines = self.engines.lock().expect("engines lock");
        engines.get(slug).and_then(|e| {
            e.ready
                .then(|| (format!("http://127.0.0.1:{}", e.port), e.context_tokens))
        })
    }

    /// Test seam: register a fake "ready" engine at an arbitrary port so
    /// integration tests can route local turns at a mock upstream without
    /// spawning a real llama-server.
    #[doc(hidden)]
    pub(crate) fn register_for_test(&self, slug: &str, port: u16) {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        self.engines.lock().expect("engines lock").insert(
            slug.to_string(),
            EngineEntry {
                port,
                context_tokens: LOCAL_CONTEXT_TOKENS,
                ready: true,
                shutdown: tx,
            },
        );
    }
}

// ============================================================================
// Pure helpers
// ============================================================================

/// Directory holding downloaded model files.
pub(crate) fn models_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("models")
}

/// Strip the `local/` prefix if present, so public APIs accept either the
/// full model id or the bare slug.
fn slug_of(id_or_slug: &str) -> &str {
    id_or_slug
        .strip_prefix(LOCAL_MODEL_PREFIX)
        .unwrap_or(id_or_slug)
}

/// Whether a model id routes to the local engine.
pub(crate) fn is_local_model(model: &str) -> bool {
    model.starts_with(LOCAL_MODEL_PREFIX)
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

/// The slug (and thus the `local/<slug>` id) for a model file name.
fn slug_for_file(file_name: &str) -> String {
    file_name
        .strip_suffix(".gguf")
        .or_else(|| file_name.strip_suffix(".GGUF"))
        .unwrap_or(file_name)
        .to_string()
}

/// A human display name derived from a file stem when no sidecar or
/// catalog entry supplies one.
fn prettify_stem(stem: &str) -> String {
    stem.replace(['-', '_'], " ")
}

/// Locate the `llama-server` binary: config override first, then `PATH`,
/// then the usual install locations (a GUI launched from Finder does not
/// inherit a shell `PATH`, so Homebrew's prefix is checked explicitly).
pub(crate) fn resolve_engine_path(cfg: &Config) -> Option<PathBuf> {
    if let Some(overridden) = cfg.llama_server_path_override.as_deref() {
        let p = PathBuf::from(overridden);
        return p.is_file().then_some(p);
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join("llama-server");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
        let candidate = Path::new(dir).join("llama-server");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
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
    /// Snapshot the whole local-inference domain: resolved engine binary +
    /// every model (on disk, downloading, loading, loaded).
    pub(crate) async fn local_models_state(&self) -> Result<LocalModelsState, AppError> {
        let cfg = self.load_config();
        let engine_path = resolve_engine_path(&cfg).map(|p| p.display().to_string());
        let dir = models_dir(&self.data_dir);

        let mut models: Vec<LocalModelInfo> = Vec::new();

        // On-disk models.
        if let Ok(mut entries) = tokio::fs::read_dir(&dir).await {
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

                let status = {
                    let engines = self.local.engines.lock().expect("engines lock");
                    match engines.get(&slug) {
                        Some(e) if e.ready => LocalModelStatus::Loaded {
                            port: e.port,
                            context_tokens: e.context_tokens,
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
                    .get(&slug)
                    .cloned();

                let display_name = sidecar
                    .as_ref()
                    .map(|s| s.display_name.clone())
                    .unwrap_or_else(|| prettify_stem(&slug));

                models.push(LocalModelInfo {
                    id: format!("{LOCAL_MODEL_PREFIX}{slug}"),
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

        // In-flight downloads (not yet on disk as `.gguf`).
        {
            let downloads = self.local.downloads.lock().expect("downloads lock");
            for (slug, dl) in downloads.iter() {
                if models.iter().any(|m| &m.slug == slug) {
                    continue;
                }
                let total = dl.total.load(Ordering::Relaxed);
                models.push(LocalModelInfo {
                    id: format!("{LOCAL_MODEL_PREFIX}{slug}"),
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

        // Failures for slugs with no file and no active download (a failed
        // download leaves nothing on disk — the error must stay visible).
        {
            let failures = self.local.failures.lock().expect("failures lock");
            for (slug, message) in failures.iter() {
                if models.iter().any(|m| &m.slug == slug) {
                    continue;
                }
                models.push(LocalModelInfo {
                    id: format!("{LOCAL_MODEL_PREFIX}{slug}"),
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
        Ok(LocalModelsState {
            engine_path,
            models,
        })
    }

    /// Start downloading a model from `url` in the background. Returns the
    /// `local/<slug>` id immediately; progress and completion arrive as
    /// [`Change::LocalModels`] emissions. If the URL matches a curated
    /// catalog entry its display name is adopted.
    pub(crate) async fn download_local_model(&self, url: &str) -> Result<String, AppError> {
        let (download_url, file_name) = normalize_model_url(url)?;
        let slug = slug_for_file(&file_name);
        let id = format!("{LOCAL_MODEL_PREFIX}{slug}");
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
        self.local
            .failures
            .lock()
            .expect("failures lock")
            .remove(&slug);
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
                local
                    .failures
                    .lock()
                    .expect("failures lock")
                    .insert(slug_task, e.to_string());
            }
            bus.emit(Change::LocalModels);
        });

        Ok(id)
    }

    /// Cancel an in-flight download. The transfer task removes the partial
    /// file and emits when it notices (within one chunk).
    pub(crate) async fn cancel_local_model_download(&self, id: &str) -> Result<(), AppError> {
        let slug = slug_of(id);
        let downloads = self.local.downloads.lock().expect("downloads lock");
        match downloads.get(slug) {
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
    /// model is loaded or loading — unload first, explicitly.
    pub(crate) async fn delete_local_model(&self, id: &str) -> Result<(), AppError> {
        let slug = slug_of(id).to_string();
        {
            let engines = self.local.engines.lock().expect("engines lock");
            if engines.contains_key(&slug) {
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
        let path = models_dir(&self.data_dir).join(format!("{slug}.gguf"));
        if path.exists() {
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
            .remove(&slug);
        self.bus.emit(Change::LocalModels);
        Ok(())
    }

    /// Load a model: spawn `llama-server` on a free loopback port and wait
    /// until its `/health` endpoint reports ready (or the load fails).
    /// Emits [`Change::LocalModels`] at spawn, on ready, and on failure; a
    /// supervisor task owns the child and also emits if the engine later
    /// exits unexpectedly.
    pub(crate) async fn load_local_model(&self, id: &str) -> Result<(), AppError> {
        let slug = slug_of(id).to_string();
        let cfg = self.load_config();

        let model_path = models_dir(&self.data_dir).join(format!("{slug}.gguf"));
        if !model_path.is_file() {
            return Err(AppError::LocalModel {
                message: format!("model `{slug}` is not downloaded"),
            });
        }
        let engine = resolve_engine_path(&cfg).ok_or_else(|| AppError::LocalModel {
            message: "llama-server not found — install llama.cpp (e.g. `brew install llama.cpp`) \
                      or set `llama_server_path` in config"
                .into(),
        })?;

        let port = pick_free_port()?;
        let model_id = format!("{LOCAL_MODEL_PREFIX}{slug}");

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

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        {
            let mut engines = self.local.engines.lock().expect("engines lock");
            if engines.contains_key(&slug) {
                return Err(AppError::LocalModel {
                    message: format!("`{slug}` is already loaded"),
                });
            }
            engines.insert(
                slug.clone(),
                EngineEntry {
                    port,
                    context_tokens: LOCAL_CONTEXT_TOKENS,
                    ready: false,
                    shutdown: shutdown_tx,
                },
            );
        }
        self.local
            .failures
            .lock()
            .expect("failures lock")
            .remove(&slug);
        self.bus.emit(Change::LocalModels);

        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
        let bus = self.bus.clone();
        let local = self.local.clone();
        let http = match &self.http_override {
            Some(c) => c.clone(),
            None => plain_http_client()?,
        };
        let slug_task = slug.clone();
        // Supervisor task: owns the child for its whole life. Cancellation
        // authority is the shutdown channel (map removal → send) — and if
        // the whole runtime is torn down, `kill_on_drop` reaps the child.
        tokio::spawn(async move {
            supervise_engine(
                command,
                port,
                http,
                shutdown_rx,
                ready_tx,
                bus,
                local,
                slug_task,
            )
            .await;
        });

        match ready_rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(message)) => Err(AppError::LocalModel { message }),
            Err(_) => Err(AppError::LocalModel {
                message: "engine supervisor exited before the model became ready".into(),
            }),
        }
    }

    /// Unload a model: signal its supervisor, which kills the subprocess.
    pub(crate) async fn unload_local_model(&self, id: &str) -> Result<(), AppError> {
        let slug = slug_of(id).to_string();
        let entry = {
            let mut engines = self.local.engines.lock().expect("engines lock");
            engines.remove(&slug)
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
                message: format!("`{slug}` is not loaded"),
            }),
        }
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
    slug: String,
) {
    let fail = |message: &str| {
        local.engines.lock().expect("engines lock").remove(&slug);
        local
            .failures
            .lock()
            .expect("failures lock")
            .insert(slug.clone(), message.to_string());
    };

    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            let message = format!("failed to start llama-server: {e}");
            fail(&message);
            bus.emit(Change::LocalModels);
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
        if let Ok(resp) = http.get(&health_url).send().await
            && resp.status().is_success()
        {
            break LoadEnd::Ready;
        }
    };

    match end {
        LoadEnd::Ready => {}
        LoadEnd::Shutdown => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            // Map entry was already removed by `unload`.
            bus.emit(Change::LocalModels);
            let _ = ready_tx.send(Err("load cancelled".into()));
            return;
        }
        LoadEnd::Exited(code) => {
            let message = format!(
                "llama-server exited during load (status {code:?}). Last output:\n{}",
                tail_text(),
            );
            fail(&message);
            bus.emit(Change::LocalModels);
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
            bus.emit(Change::LocalModels);
            let _ = ready_tx.send(Err(message));
            return;
        }
    }

    // Ready — flip the map entry (it may have been removed by a concurrent
    // unload, in which case we shut down instead of serving a ghost). The
    // decision happens under the lock; the kill happens after it drops.
    let still_wanted = {
        let mut engines = local.engines.lock().expect("engines lock");
        match engines.get_mut(&slug) {
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
    bus.emit(Change::LocalModels);
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
            bus.emit(Change::LocalModels);
        }
        Some(exit) => {
            let code = exit.ok().and_then(|s| s.code());
            let message = format!(
                "llama-server exited unexpectedly (status {code:?}). Last output:\n{}",
                tail_text(),
            );
            fail(&message);
            bus.emit(Change::LocalModels);
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
        let id = format!("{LOCAL_MODEL_PREFIX}{slug}");
        assert_eq!(slug_of(&id), slug);
        assert_eq!(slug_of(&slug), slug);
        assert!(is_local_model(&id));
        assert!(!is_local_model("gemma4-31b"));
    }
}
