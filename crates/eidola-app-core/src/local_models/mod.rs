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
//!
//! ## Layout
//!
//! This module is what every build shares: the curated catalog, the
//! snapshot types, download-URL resolution, and the plain HTTP clients
//! engines and external backends are reached through. The engines
//! themselves — the runtime, downloads, subprocess supervision — are
//! `engine`, behind the `desktop` feature. Without it, `unavailable` stands
//! in: no engine is ever registered, so the chat path's lease finds nothing
//! and its load is refused with [`AppError::LocalModel`].

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::transport::{Redirects, Transport};

/// Context window requested from `llama-server` (`-c`). Deliberately
/// bounded — KV-cache memory scales with it — and far below what Gemma 4
/// supports; a later change can make this per-model configuration.
#[cfg(any(feature = "desktop", feature = "test-support"))]
pub(crate) const LOCAL_CONTEXT_TOKENS: u32 = 8192;

#[cfg(feature = "desktop")]
mod engine;
#[cfg(feature = "desktop")]
pub(crate) use engine::*;

#[cfg(not(feature = "desktop"))]
mod unavailable;
#[cfg(not(feature = "desktop"))]
pub(crate) use unavailable::{EngineLease, LocalRuntime};

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

impl LocalCatalogEntry {
    /// The `<slug>@local` model reference this entry is selected by once it
    /// is in the managed store — the slug is its file's stem, so the
    /// reference is spelled from `file_name`, never from `id`.
    pub fn model_ref(&self) -> String {
        engine_model_id(
            crate::backends::LOCAL_BACKEND_ID,
            &slug_for_file(self.file_name),
        )
    }
}

/// The curated entry a model reference names in the managed local store, if
/// any.
pub fn catalog_entry_for_model_ref(model_ref: &str) -> Option<&'static LocalCatalogEntry> {
    LOCAL_MODEL_CATALOG
        .iter()
        .find(|entry| entry.model_ref() == model_ref)
}

// ============================================================================
// DTOs — snapshots returned to the CLI/GUI.
// ============================================================================

/// Lifecycle state of one local model, merged from the filesystem and the
/// runtime maps.
/// **Serialized by the local control protocol** ([`crate::ipc`]): a field
/// rename here is a wire change.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
/// **Serialized by the local control protocol** ([`crate::ipc`]): a field
/// rename here is a wire change.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    /// True when the directory scan found this row's `.gguf`.
    ///
    /// A snapshot's rows do not all come from the filesystem: the managed
    /// store *synthesizes* a row for an in-flight download (which streams to
    /// `<file>.part` and only becomes a `.gguf` at the final rename) and for
    /// a standing failure with nothing left on disk, so a re-download of a
    /// slug whose file was deleted out of band puts a row in the snapshot
    /// while the file is gone. Any caller reasoning about *file presence* —
    /// rather than about what to show the user — must read this, not the
    /// row's existence or its status.
    pub on_disk: bool,
}

/// Snapshot of the whole local-inference domain: the managed `local`
/// singleton plus every configured `llamacpp` backend's scanned directory.
/// **Serialized by the local control protocol** ([`crate::ipc`]): a field
/// rename here is a wire change.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

/// One live engine, straight from the in-process registry.
///
/// This is deliberately **not** a [`LocalModelInfo`]: it carries only what
/// the registry itself knows, and nothing that came from a directory scan.
/// There is no `display_name` here because the registry has none — the
/// pretty name lives in a `.meta.json` sidecar beside the `.gguf`, and the
/// whole point of this type is to describe an engine whose file may be gone.
/// A caller that wants a nice name joins this to
/// [`LocalModelsState`] on `id` and falls back to `slug`.
/// **Serialized by the local control protocol** ([`crate::ipc`]): a field
/// rename here is a wire change.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunningEngine {
    /// The chat-routable selection id, `<slug>@<backend-id>` — the join key
    /// against [`LocalModelInfo::id`].
    pub id: String,
    pub backend_id: String,
    /// The file stem the engine was loaded from. The truest name we still
    /// have when the file itself has gone.
    pub slug: String,
    /// The loopback port the engine serves on.
    pub port: u16,
    pub context_tokens: u32,
    /// False while the engine is still warming (the `Loading` state).
    pub ready: bool,
    pub pinned: bool,
}

/// One `llamacpp` backend's scanned models.
/// **Serialized by the local control protocol** ([`crate::ipc`]): a field
/// rename here is a wire change.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

// ============================================================================
// Shared helpers
// ============================================================================

/// What retiring a backend actually did.
///
/// Retirement has **two** effects — it stops that backend's engines and it
/// forgets what those engines had to say — and a caller deciding whether
/// anything needs re-rendering has to hear about both. (It has a third: the
/// backend's `BackendEpoch` moves, invalidating loads that are still
/// in flight. That one is deliberately not summarised here, because it
/// changes nothing anyone is rendering — a load far enough along to be
/// visible has an entry, and entries are counted.) A count of engines was
/// an honest summary while stopping them was the only effect; the moment
/// forgetting reports joined it, that count started answering a different
/// question than the one its caller was asking, and a backend whose engine had
/// already failed (a standing report, nothing running) retired in silence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Retirement {
    /// Engine supervisors signalled to stop.
    pub(crate) engines: usize,
    /// Standing engine reports forgotten.
    pub(crate) reports: usize,
}

impl Retirement {
    /// Whether anything a local-models snapshot renders actually changed.
    /// `false` is a genuine no-op — nothing was running and nothing was being
    /// reported — and its caller must stay silent, because an invalidation
    /// nobody needs is its own defect.
    pub(crate) fn changed(self) -> bool {
        self.engines > 0 || self.reports > 0
    }
}

/// The selectable model id for an engine-backed backend's slug — the
/// uniform qualified form ([`crate::backends::qualified_model_id`]). The id
/// doubles as the engine's `--alias`, so the wire model in a chat body
/// equals the selection string.
pub(crate) fn engine_model_id(backend_id: &str, slug: &str) -> String {
    crate::backends::qualified_model_id(slug, backend_id)
}

/// What a model URL resolves to: where the bytes come from, the file they
/// land in, and the **slug that identifies the transfer**.
///
/// The slug is the identity [`AppCore::download_local_model`] deduplicates on,
/// so it is also the only honest key for a caller that wants to know whether a
/// transfer is already under way: equivalent spellings of one model (a Hugging
/// Face `/blob/` page versus its `/resolve/` object, a `?download=true`
/// suffix, any two URLs naming the same file) collapse onto it exactly as they
/// collapse in app-core's own map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelDownloadTarget {
    /// The URL the bytes are actually fetched from: the caller's own URL,
    /// query string included, with a Hugging Face `/blob/` page rewritten to
    /// its `/resolve/` object.
    ///
    /// **The query is not decoration.** A direct link to a model on S3 or a
    /// CDN commonly carries its authorization in the query
    /// (`?token=…`, `?X-Amz-Signature=…`), so a request made without it is a
    /// request that fails — while the *path* is all that identifies the file.
    /// Hence the split: inspection and identity below read the path, and only
    /// this field is handed to the HTTP client.
    pub url: String,
    /// The same URL with query and fragment removed — what is recorded beside
    /// the file and shown as its source. Provenance, never a request: a
    /// signed URL's token is a short-lived credential that has no business
    /// being written to disk, and it says nothing about where the model came
    /// from that the path does not.
    pub canonical_url: String,
    /// The file name they are written under.
    pub file_name: String,
    /// The `<slug>` of the resulting `<slug>@local` id — the transfer's identity.
    pub slug: String,
}

/// Resolve a pasted or curated model URL to its [`ModelDownloadTarget`].
///
/// **The single derivation of a download's identity.** `download_local_model`
/// resolves through it, and so must anything outside app-core that keys work
/// on "this download" — a caller that keys on the raw URL instead lets one
/// transfer be requested twice under two spellings, which is the duplicate
/// app-core would then refuse.
pub fn resolve_model_download(input: &str) -> Result<ModelDownloadTarget, AppError> {
    let (url, file_name) = normalize_model_url(input)?;
    let slug = slug_for_file(&file_name);
    Ok(ModelDownloadTarget {
        canonical_url: strip_query(&url).to_string(),
        url,
        file_name,
        slug,
    })
}

/// A URL without its query or fragment — the part that names the file.
fn strip_query(url: &str) -> &str {
    let without_fragment = url.split('#').next().unwrap_or(url);
    without_fragment
        .split('?')
        .next()
        .unwrap_or(without_fragment)
}

/// Normalize a pasted model URL into `(download_url, file_name)`.
///
/// Accepts direct `.gguf` URLs and Hugging Face `/blob/` page URLs (the
/// address bar of a file page), which are rewritten to `/resolve/` so they
/// download instead of rendering HTML. Anything without a `.gguf` final
/// path segment is rejected — better an honest error at paste time than a
/// half-downloaded HTML page named like a model.
///
/// **The query is stripped for inspection, not from the download.** What the
/// file is called, whether it is a `.gguf` at all, and which transfer it
/// identifies are all decided on the path alone, so every equivalent spelling
/// of one model still collapses onto one `file_name` (and one slug) —
/// the identity `download_local_model` and its callers key on. But the URL
/// returned here is the one that gets requested, and dropping its query would
/// strip the authorization a signed S3/CDN link carries, turning a link that
/// validates into a download that 403s. The fragment is dropped outright: it
/// is never sent to a server.
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
    // Split off the query (e.g. `?download=true`, or a signed link's
    // credentials) and drop the fragment. Everything below inspects the path;
    // the query rides along to the request untouched.
    let without_fragment = raw.split('#').next().unwrap_or(raw);
    let (path_part, query) = match without_fragment.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (without_fragment, None),
    };

    // Hugging Face file-page URLs use `/blob/<rev>/...`; the raw object
    // lives at `/resolve/<rev>/...`.
    let path = if path_part.contains("huggingface.co/") {
        path_part.replacen("/blob/", "/resolve/", 1)
    } else {
        path_part.to_string()
    };

    let file_name = path.rsplit('/').next().unwrap_or_default().to_string();
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
    let url = match query {
        Some(query) => format!("{path}?{query}"),
        None => path,
    };
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

/// A plain (non-attesting) HTTPS-capable client: native trust roots, used
/// for model downloads and loopback engine traffic.
///
/// **Follows redirects, deliberately**: a model download is a `GET` for public
/// bytes whose integrity is checked afterwards, and every large-file host
/// redirects to a CDN. See [`proxy_http_client`] for the client that does not,
/// and why the difference is about what the request *carries*.
pub(crate) fn plain_http_client(
    transport: &Transport,
) -> Result<crate::transport::Client, AppError> {
    crate::transport::plain_client(
        transport,
        Some(concat!("eidola-app-core/", env!("CARGO_PKG_VERSION"))),
        Redirects::Follow,
    )
}

/// The client the **local inference proxy's** own upstream requests go out on.
///
/// Two differences from [`plain_http_client`], and both are about the fact that
/// a proxied request carries somebody's prompt to a destination the reader
/// ticked by name:
///
/// - **No `User-Agent`.** The proxy's upstream header set is an enumerated
///   allowlist whose "and nothing else" is a claim the Record repeats back to
///   the reader (`proxy::route::UpstreamHeaders::for_record`), so a header the
///   *builder* adds would travel on every proxied completion — a version
///   fingerprint on an external backend's wire — while the row said it did not.
/// - **No redirects.** reqwest's default follows up to ten and replays a
///   cloneable body on `307`/`308`, so an exposed external backend could answer
///   `/v1/chat/completions` with a `Location` pointing anywhere and receive the
///   whole prompt at an origin the reader never exposed — while the Record went
///   on naming the configured backend, because that is the backend the request
///   was made to. reqwest strips a cross-origin `Authorization`; it does not
///   strip the body, and the body is the sensitive part here. A redirect is
///   therefore handed back as the answer it is: the caller sees the upstream's
///   own `3xx`, the Record shows it, and nothing of the reader's leaves for a
///   destination they did not choose.
///
/// The app's own completions refuse redirects for the same reason
/// ([`completion_http_client`]), and so does the attested client
/// (`tinfoil_verifier::attesting_client`).
#[cfg(feature = "desktop")]
pub(crate) fn proxy_http_client(
    transport: &Transport,
) -> Result<crate::transport::Client, AppError> {
    crate::transport::plain_client(transport, None, Redirects::Refuse)
}

/// The client a **chat turn or a chore** sends its completion on, to a local
/// engine or an external OpenAI-compatible backend.
///
/// **No redirects**, for the proxy's reason ([`proxy_http_client`]): a turn's
/// body is the conversation, and a `307`/`308` would have reqwest replay it to
/// whatever origin the `Location` names — one the reader never configured,
/// while the Record goes on naming the backend they did. A redirect is handed
/// back as the answer it is, and the turn fails on the upstream's own status.
/// It keeps [`plain_http_client`]'s `User-Agent`: the turn path's header set is
/// not an allowlist the Record repeats back, which is the proxy's reason for
/// dropping it.
pub(crate) fn completion_http_client(
    transport: &Transport,
) -> Result<crate::transport::Client, AppError> {
    crate::transport::plain_client(
        transport,
        Some(concat!("eidola-app-core/", env!("CARGO_PKG_VERSION"))),
        Redirects::Refuse,
    )
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
    fn normalize_rewrites_hf_blob_urls_and_keeps_the_query() {
        let (url, name) = normalize_model_url(
            "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/blob/main/gemma-4-12b-it-qat-q4_0.gguf?download=true#anchor",
        )
        .unwrap();
        assert_eq!(
            url,
            "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/gemma-4-12b-it-qat-q4_0.gguf?download=true",
            "the page URL is rewritten to the object; the query rides along"
        );
        assert_eq!(name, "gemma-4-12b-it-qat-q4_0.gguf");
    }

    /// A direct link's query is its **authorization**, not decoration: signed
    /// S3/CDN model links carry `?token=…`, and a request made without it is a
    /// request that 403s. So the query is stripped for *inspection* — the file
    /// name, the `.gguf` check, the transfer's identity — and kept on the URL
    /// that is actually fetched.
    #[test]
    fn a_signed_urls_authorization_survives_to_the_fetch() {
        let signed = "https://cdn.example.com/models/tiny.gguf?token=abc123&X-Amz-Expires=900";
        let target = resolve_model_download(signed).expect("resolves");
        assert_eq!(target.url, signed, "the fetch keeps every query parameter");
        assert_eq!(target.file_name, "tiny.gguf");
        assert_eq!(target.slug, "tiny");
        assert_eq!(
            target.canonical_url, "https://cdn.example.com/models/tiny.gguf",
            "provenance carries no credential"
        );
    }

    /// Equivalent spellings of one model resolve to one transfer identity.
    /// Callers key work on that slug — a key taken from the raw text would let
    /// the same transfer be requested twice, once per spelling.
    #[test]
    fn equivalent_spellings_resolve_to_one_slug() {
        let base = "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf";
        let file = "gemma-4-12b-it-qat-q4_0.gguf";
        let spellings = [
            format!("{base}/resolve/main/{file}"),
            format!("{base}/blob/main/{file}"),
            format!("{base}/blob/main/{file}?download=true"),
            format!("{base}/resolve/main/{file}#fragment"),
            format!("  {base}/resolve/main/{file}  "),
        ];
        let first = resolve_model_download(&spellings[0]).expect("resolves");
        assert_eq!(first.slug, "gemma-4-12b-it-qat-q4_0");
        for spelling in &spellings[1..] {
            let target = resolve_model_download(spelling).expect("resolves");
            assert_eq!(
                (&target.slug, &target.file_name, &target.canonical_url),
                (&first.slug, &first.file_name, &first.canonical_url),
                "`{spelling}` names the same transfer"
            );
        }
        // Identity is the path's alone, so two spellings that differ *only*
        // in their query — one signed link and its refreshed twin — are still
        // one transfer, and the second is refused rather than started again.
        let base = format!("{base}/resolve/main/{file}");
        let signed = resolve_model_download(&format!("{base}?token=one")).expect("resolves");
        let refreshed = resolve_model_download(&format!("{base}?token=two")).expect("resolves");
        assert_eq!(signed.slug, refreshed.slug);
        assert_ne!(signed.url, refreshed.url, "each fetches with its own token");
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
    fn default_model_is_the_curated_e4b_in_the_local_store() {
        let entry = catalog_entry_for_model_ref(crate::config::DEFAULT_MODEL)
            .expect("the default model must be a curated local catalog entry");
        assert_eq!(entry.id, "gemma-4-e4b");
        assert_eq!(entry.model_ref(), crate::config::DEFAULT_MODEL);
    }

    #[test]
    fn a_curated_models_default_label_is_its_display_name() {
        for entry in LOCAL_MODEL_CATALOG {
            assert_eq!(
                crate::db::default_agent_label(&entry.model_ref()),
                entry.display_name
            );
        }
        // Anything else is title-cased by token, backend dropped.
        assert_eq!(crate::db::default_agent_label("qwen3-8b@ext"), "Qwen3 8b");
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
}
