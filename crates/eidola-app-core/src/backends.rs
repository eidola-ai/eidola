//! The backend registry — *where an ask can be routed*.
//!
//! ## Design
//!
//! A **backend** is a configured inference destination. Four kinds exist
//! today, in decreasing expected frequency of use:
//!
//! - `eidola` — the confidential Eidola service (singleton). Attested
//!   transport, anonymous-credential billing. A backend row describes how
//!   to reach and trust that backend, so the eidola row owns the whole
//!   connection + trust bundle: its `base_url`, `trusted_measurements`
//!   (JSON overrides), and `hardware_root_ca` / `hardware_intermediate_ca`
//!   (PEM ARK/ASK overrides). Each is NULL by default, which means "use the
//!   embedded trust-root pin baked into this build"; `update_backend`
//!   accepts exactly this bundle for the eidola row (and nothing else).
//! - `local` — Eidola-managed on-device models (singleton): the curated
//!   catalog and downloads under `<data_dir>/models`, served by managed
//!   llama.cpp engines. Gated by device capability, no account required.
//! - `openai` — any OpenAI-compatible HTTP server the user configures:
//!   a self-hosted vLLM/llama.cpp/Ollama box, or a conventional provider
//!   the user chooses to trust without confidential-computing guarantees.
//!   Plain HTTPS + optional bearer key; no credential spend.
//! - `llamacpp` — a user-owned llama.cpp install: Eidola scans the
//!   backend's `models_dir` and starts/stops `llama-server` engines on
//!   demand, but does **not** manage (download/delete) the model files.
//!
//! The planned self-hosted-device routing (an Eidola instance on your own
//! Mac serving your phone) slots in as a future kind: one enum variant, one
//! transport arm in `prepare_turn`, the same row shape — the registry,
//! model references, picker grouping, and forensics all already fit it.
//!
//! ## Model references
//!
//! One rule, uniformly applied: a model selection is
//! `<model>@<backend-id>`, parsed at the **last** `@` (backend ids are
//! validated to `[a-z0-9-]`, so the split is unambiguous even for
//! Ollama-style `name:tag` or HF-style `org/model` ids). The single sugar:
//! `eidola`, being the default backend, may be written bare — `gemma4-31b`
//! means `gemma4-31b@eidola`, and [`qualified_model_id`] keeps the bare
//! form canonical so the common case reads clean everywhere (config,
//! action rows, the CLI). Every other backend — the local singleton
//! included — always spells its models qualified (`<slug>@local`).
//!
//! ## Forensics
//!
//! `request.backend_id` records which backend serviced each request. Rows
//! are soft-removed (`removed_at`), never deleted, so that reference stays
//! resolvable forever; re-adding a backend with the same id revives the row.

use serde::{Deserialize, Serialize};

use crate::changes::Change;
use crate::db;
use crate::error::AppError;
use crate::{Inner, ModelInfo, now_ms};

/// The reserved singleton backend ids.
pub const EIDOLA_BACKEND_ID: &str = "eidola";
pub const LOCAL_BACKEND_ID: &str = "local";

/// The kind of a configured backend. See the module docs for semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// The confidential Eidola service (singleton).
    Eidola,
    /// Eidola-managed on-device models (singleton).
    Local,
    /// A user-configured OpenAI-compatible server.
    OpenAi,
    /// A user-owned llama.cpp install whose engines Eidola manages.
    LlamaCpp,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Eidola => "eidola",
            BackendKind::Local => "local",
            BackendKind::OpenAi => "openai",
            BackendKind::LlamaCpp => "llamacpp",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "eidola" => Some(BackendKind::Eidola),
            "local" => Some(BackendKind::Local),
            "openai" => Some(BackendKind::OpenAi),
            "llamacpp" => Some(BackendKind::LlamaCpp),
            _ => None,
        }
    }

    /// Whether this kind serves models through locally managed llama.cpp
    /// engines (vs. a remote catalog fetch).
    pub fn is_engine_backed(self) -> bool {
        matches!(self, BackendKind::Local | BackendKind::LlamaCpp)
    }
}

/// One configured backend, as surfaced to the CLI/GUI. The api key itself
/// is never carried — only whether one is set (replacing it is write-only).
#[derive(Clone, Debug)]
pub struct BackendInfo {
    pub id: String,
    pub kind: BackendKind,
    pub display_name: String,
    pub enabled: bool,
    pub base_url: Option<String>,
    pub has_api_key: bool,
    pub models_dir: Option<String>,
    /// Manually pinned model list; `None` = trust the backend's listing.
    pub model_overrides: Option<Vec<String>>,
    /// `llamacpp` only: explicit `llama-server` path; `None` = discover it.
    pub engine_path: Option<String>,
    /// `llamacpp` only: may a request auto-start an engine on demand? The
    /// `local` backend always auto-starts (it's Eidola's own engine).
    pub auto_start: bool,
    pub created_at: i64,
}

impl BackendInfo {
    pub(crate) fn from_row(row: &db::BackendRow) -> Result<Self, AppError> {
        let kind = BackendKind::parse(&row.kind).ok_or_else(|| AppError::Database {
            message: format!("unknown backend kind `{}` for `{}`", row.kind, row.id),
        })?;
        Ok(BackendInfo {
            id: row.id.clone(),
            kind,
            display_name: row.display_name.clone(),
            enabled: row.enabled,
            base_url: row.base_url.clone(),
            has_api_key: row.api_key.is_some(),
            models_dir: row.models_dir.clone(),
            model_overrides: parse_overrides(row.model_overrides.as_deref())?,
            engine_path: row.engine_path.clone(),
            auto_start: row.auto_start,
            created_at: row.created_at,
        })
    }
}

/// A parsed model selection: which backend, and the model id the backend
/// itself understands (the *wire* model).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRef {
    pub backend_id: String,
    pub model: String,
}

/// Parse a model selection string: `<model>@<backend>` splits at the last
/// `@`; a bare selection routes to `eidola` (the default backend's sugar).
/// Pure and infallible — whether the backend *exists* is the router's
/// question, not the parser's.
pub fn parse_model_ref(selection: &str) -> ModelRef {
    if let Some(at) = selection.rfind('@') {
        let (model, backend) = selection.split_at(at);
        let backend = &backend[1..];
        if !model.is_empty() && !backend.is_empty() {
            return ModelRef {
                backend_id: backend.to_string(),
                model: model.to_string(),
            };
        }
    }
    ModelRef {
        backend_id: EIDOLA_BACKEND_ID.to_string(),
        model: selection.to_string(),
    }
}

/// The canonical selection string for a (model, backend) pair — the inverse
/// of [`parse_model_ref`]. Eidola models stay bare (the default backend's
/// sugar); every other backend's models are spelled qualified.
pub fn qualified_model_id(model: &str, backend_id: &str) -> String {
    if backend_id == EIDOLA_BACKEND_ID {
        model.to_string()
    } else {
        format!("{model}@{backend_id}")
    }
}

/// Validate a user-chosen backend id: lowercase alphanumerics and hyphens,
/// 1–32 chars, not a reserved singleton id. The character set is what makes
/// the `@` split in [`parse_model_ref`] unambiguous.
pub fn validate_backend_id(id: &str) -> Result<(), AppError> {
    if id.is_empty() || id.len() > 32 {
        return Err(AppError::Config {
            message: "backend id must be 1–32 characters".into(),
        });
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(AppError::Config {
            message: "backend id may contain only lowercase letters, digits, and hyphens".into(),
        });
    }
    if id == EIDOLA_BACKEND_ID || id == LOCAL_BACKEND_ID {
        return Err(AppError::Config {
            message: format!("`{id}` is a reserved backend id"),
        });
    }
    Ok(())
}

fn parse_overrides(raw: Option<&str>) -> Result<Option<Vec<String>>, AppError> {
    match raw {
        None => Ok(None),
        Some(text) => serde_json::from_str::<Vec<String>>(text)
            .map(Some)
            .map_err(|e| AppError::Database {
                message: format!("invalid model_overrides JSON: {e}"),
            }),
    }
}

fn overrides_to_json(overrides: Option<&[String]>) -> Option<String> {
    overrides.map(|list| serde_json::to_string(list).expect("string list serializes"))
}

/// Fields for adding (or reviving) an external backend. Kind-specific
/// requirements are validated in [`Inner::add_backend`].
#[derive(Clone, Debug)]
pub struct NewBackend {
    pub id: String,
    pub kind: BackendKind,
    /// Display name; defaults to the id when empty.
    pub display_name: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub models_dir: Option<String>,
    pub model_overrides: Option<Vec<String>>,
    /// `llamacpp` only: explicit `llama-server` path (`None` = discover it).
    pub engine_path: Option<String>,
    /// `llamacpp` only: whether a request may auto-start an engine. Ignored
    /// (and required `true`) for other kinds.
    pub auto_start: bool,
}

/// Partial update of an external backend's configuration. `None` leaves the
/// field alone; `Some(None)` clears a nullable field.
#[derive(Clone, Debug, Default)]
pub struct BackendUpdate {
    pub display_name: Option<String>,
    pub base_url: Option<Option<String>>,
    pub api_key: Option<Option<String>>,
    pub models_dir: Option<Option<String>>,
    pub model_overrides: Option<Option<Vec<String>>>,
    /// `llamacpp` only: set/clear the explicit engine path.
    pub engine_path: Option<Option<String>>,
    /// `llamacpp` only: flip request-triggered auto-start.
    pub auto_start: Option<bool>,
    /// `eidola` only: set/clear the enclave-measurement override list (a JSON
    /// array of the `EnclaveMeasurement` shape). `Some(None)` reverts to pin.
    pub trusted_measurements: Option<Option<String>>,
    /// `eidola` only: set/clear the PEM ARK certificate override.
    pub hardware_root_ca: Option<Option<String>>,
    /// `eidola` only: set/clear the PEM ASK certificate override.
    pub hardware_intermediate_ca: Option<Option<String>>,
}

// ============================================================================
// Inner methods — the operations `AppCore` wraps.
// ============================================================================

impl Inner {
    /// Hold this backend's configuration gate for the whole of one
    /// configuration operation — **its database write and the cleanup that
    /// belongs to it**.
    ///
    /// Those two steps are one operation, and nothing but a critical section
    /// makes that true. Each of these methods commits and then acts on
    /// in-memory state (retiring engines and their reports), so without the
    /// gate an older operation's cleanup can arrive after a newer write has
    /// already settled the row: disable commits, enable commits over it, a
    /// load registers the engine the *final* configuration authorises, and the
    /// disable's cleanup then stops it — a load that reported success, an
    /// enabled backend, and no engine.
    ///
    /// **An in-memory generation cannot replace this.** The cleanup that
    /// should run is the one belonging to the write the database kept, and a
    /// counter bumped either side of the commit orders itself, not the
    /// commits: two operations can bump in one order and commit in the other.
    /// Ordering a generation against the commit means writing it *in* the
    /// transaction — a persisted column, which is a schema change — or
    /// serializing, which is this.
    ///
    /// Holding a lock across a load's work was refused for good reason (an
    /// engine load walks the filesystem, picks a port and spawns a child, and
    /// a configuration edit must not wait on any of that). The reasoning does
    /// not carry here: a configuration operation is one short database
    /// statement plus a synchronous sweep of two in-memory maps, it touches no
    /// filesystem and no subprocess, and the gate is per backend. Loads do not
    /// take it at all — a load that read a stale row is a different axis,
    /// closed by [`crate::local_models::BackendEpoch`].
    async fn lock_backend_config(&self, id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let gate = {
            let mut gates = self
                .backend_config_gates
                .lock()
                .expect("backend config gates");
            gates.entry(id.to_string()).or_default().clone()
        };
        gate.lock_owned().await
    }

    pub(crate) async fn list_backends(&self) -> Result<Vec<BackendInfo>, AppError> {
        let conn = self.db_conn().await?;
        let rows = db::list_backends(&conn).await?;
        rows.iter().map(BackendInfo::from_row).collect()
    }

    /// The live, *enabled* backend a turn may route through. Distinguishes
    /// the three refusals honestly: unknown, removed, disabled.
    pub(crate) async fn require_backend(
        &self,
        conn: &turso::Connection,
        id: &str,
    ) -> Result<db::BackendRow, AppError> {
        let row = db::get_backend(conn, id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("no backend named `{id}` is configured"),
            })?;
        if row.removed_at.is_some() {
            return Err(AppError::NotConfigured {
                message: format!("backend `{id}` was removed"),
            });
        }
        if !row.enabled {
            return Err(AppError::NotConfigured {
                message: format!("backend `{id}` is disabled"),
            });
        }
        Ok(row)
    }

    /// Add an external backend (kind `openai` or `llamacpp`), or revive a
    /// soft-removed row with the same id. Emits [`Change::Backends`].
    pub(crate) async fn add_backend(&self, new: NewBackend) -> Result<BackendInfo, AppError> {
        validate_backend_id(&new.id)?;
        // Taken after validation, so a rejected id allocates no gate. A revive
        // is a configuration write like any other: it must not commit inside
        // another operation's write-and-cleanup.
        let _config = self.lock_backend_config(&new.id).await;
        match new.kind {
            BackendKind::Eidola | BackendKind::Local => {
                return Err(AppError::Config {
                    message: "the eidola and local backends are built in".into(),
                });
            }
            BackendKind::OpenAi => {
                let url = new.base_url.as_deref().unwrap_or("").trim();
                if !(url.starts_with("http://") || url.starts_with("https://")) {
                    return Err(AppError::Config {
                        message: "an OpenAI-compatible backend needs a base URL \
                                  (http:// or https://)"
                            .into(),
                    });
                }
                // `engine_path` / `auto_start` are engine-backed concerns.
                if new
                    .engine_path
                    .as_deref()
                    .is_some_and(|p| !p.trim().is_empty())
                    || !new.auto_start
                {
                    return Err(AppError::Config {
                        message: "engine path and auto-start apply only to llama.cpp backends"
                            .into(),
                    });
                }
            }
            BackendKind::LlamaCpp => {
                let dir = new.models_dir.as_deref().unwrap_or("").trim();
                if dir.is_empty() {
                    return Err(AppError::Config {
                        message: "a llama.cpp backend needs a models directory".into(),
                    });
                }
            }
        }

        let now = now_ms();
        let display_name = if new.display_name.trim().is_empty() {
            new.id.clone()
        } else {
            new.display_name.trim().to_string()
        };
        // Only `llamacpp` carries an engine path / auto-start flag; every
        // other kind stores `None` + the always-on default.
        let (engine_path, auto_start) = match new.kind {
            BackendKind::LlamaCpp => (
                new.engine_path
                    .map(|p| p.trim().to_string())
                    .filter(|p| !p.is_empty()),
                new.auto_start,
            ),
            _ => (None, true),
        };
        let row = db::BackendRow {
            id: new.id.clone(),
            kind: new.kind.as_str().to_string(),
            display_name,
            enabled: true,
            base_url: new
                .base_url
                .map(|u| u.trim().trim_end_matches('/').to_string()),
            api_key: new.api_key.filter(|k| !k.trim().is_empty()),
            models_dir: new.models_dir.map(|d| d.trim().to_string()),
            model_overrides: overrides_to_json(new.model_overrides.as_deref()),
            engine_path,
            auto_start,
            // The connection + trust bundle is the eidola row's alone; external
            // backends never carry it.
            trusted_measurements: None,
            hardware_root_ca: None,
            hardware_intermediate_ca: None,
            created_at: now,
            updated_at: now,
            removed_at: None,
        };
        let conn = self.db_conn().await?;
        db::insert_backend(&conn, &row).await?;
        self.bus.emit(Change::Backends);
        BackendInfo::from_row(&row)
    }

    /// Enable/disable any backend, singletons included — disabling `eidola`
    /// is exactly the "no account, on-device only" configuration.
    pub(crate) async fn set_backend_enabled(
        &self,
        id: &str,
        enabled: bool,
    ) -> Result<(), AppError> {
        let _config = self.lock_backend_config(id).await;
        let conn = self.db_conn().await?;
        if !db::set_backend_enabled(&conn, id, enabled, now_ms()).await? {
            return Err(AppError::NotConfigured {
                message: format!("no backend named `{id}` is configured"),
            });
        }
        if !enabled {
            // A disabled backend cannot serve a turn, so nothing may keep
            // running under its id.
            self.retire_engines_for(id).await;
        }
        self.bus.emit(Change::Backends);
        Ok(())
    }

    /// Update a backend's configuration. Per-kind field validation: the
    /// `eidola` row accepts exactly its connection + trust bundle
    /// (base_url / trusted_measurements / hardware CAs — each clearable back
    /// to NULL = the embedded pin); `local` is built in and accepts nothing;
    /// `openai` / `llamacpp` accept their own fields but never the trust
    /// bundle. Emits [`Change::Backends`] on success.
    pub(crate) async fn update_backend(
        &self,
        id: &str,
        update: BackendUpdate,
    ) -> Result<(), AppError> {
        // Held from the read: the validation below is against this row, and a
        // write landing between the two would be validated against neither.
        let _config = self.lock_backend_config(id).await;
        let conn = self.db_conn().await?;
        let row = db::get_backend(&conn, id)
            .await?
            .filter(|r| r.removed_at.is_none())
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("no backend named `{id}` is configured"),
            })?;
        let kind = BackendKind::parse(&row.kind).ok_or_else(|| AppError::Database {
            message: format!("unknown backend kind `{}`", row.kind),
        })?;

        let touches_trust = update.trusted_measurements.is_some()
            || update.hardware_root_ca.is_some()
            || update.hardware_intermediate_ca.is_some();
        let touches_external = update.display_name.is_some()
            || update.api_key.is_some()
            || update.models_dir.is_some()
            || update.model_overrides.is_some()
            || update.engine_path.is_some()
            || update.auto_start.is_some();

        match kind {
            BackendKind::Local => {
                return Err(AppError::Config {
                    message: format!("backend `{id}` is built in — only enable/disable applies"),
                });
            }
            BackendKind::Eidola => {
                // The eidola row is the confidential service's connection +
                // trust bundle — nothing else lives on it.
                if touches_external {
                    return Err(AppError::Config {
                        message: "the eidola backend accepts only connection and trust settings \
                                  (base URL, trusted measurements, hardware CAs)"
                            .into(),
                    });
                }
                if let Some(Some(url)) = update.base_url.as_ref() {
                    let url = url.trim();
                    if !(url.starts_with("http://") || url.starts_with("https://")) {
                        return Err(AppError::Config {
                            message: "base URL must start with http:// or https://".into(),
                        });
                    }
                }
            }
            BackendKind::OpenAi | BackendKind::LlamaCpp => {
                // The trust bundle is the eidola row's alone.
                if touches_trust {
                    return Err(AppError::Config {
                        message: "trusted measurements and hardware CAs apply only to the eidola \
                                  backend"
                            .into(),
                    });
                }
                // `engine_path` / `auto_start` are engine-backed concerns;
                // refuse them on an `openai` backend rather than silently
                // storing an inert value.
                if (update.engine_path.is_some() || update.auto_start.is_some())
                    && kind != BackendKind::LlamaCpp
                {
                    return Err(AppError::Config {
                        message: "engine path and auto-start apply only to llama.cpp backends"
                            .into(),
                    });
                }
            }
        }

        // Repointing a backend at another models directory (or another
        // `llama-server`) changes what `<slug>@<id>` *means*, so the engines
        // already running under that id belong to a configuration that is
        // about to stop existing — see [`Self::retire_backend_engines`].
        let repointed = update
            .models_dir
            .as_ref()
            .is_some_and(|d| d.as_deref() != row.models_dir.as_deref())
            || update.engine_path.as_ref().is_some_and(|p| {
                p.as_deref().map(|p| p.trim()).filter(|p| !p.is_empty())
                    != row.engine_path.as_deref()
            });

        let overrides_json = update
            .model_overrides
            .map(|o| overrides_to_json(o.as_deref()));
        let updated = db::update_backend_config(
            &conn,
            id,
            update.display_name.as_deref(),
            update
                .base_url
                .as_ref()
                .map(|o| o.as_deref().map(|u| u.trim().trim_end_matches('/'))),
            update.api_key.as_ref().map(|o| o.as_deref()),
            update.models_dir.as_ref().map(|o| o.as_deref()),
            overrides_json.as_ref().map(|o| o.as_deref()),
            update
                .engine_path
                .as_ref()
                .map(|o| o.as_deref().map(|p| p.trim()).filter(|p| !p.is_empty())),
            update.auto_start,
            update
                .trusted_measurements
                .as_ref()
                .map(|o| o.as_deref().filter(|m| !m.is_empty())),
            update.hardware_root_ca.as_ref().map(|o| o.as_deref()),
            update
                .hardware_intermediate_ca
                .as_ref()
                .map(|o| o.as_deref()),
            now_ms(),
        )
        .await?;
        if !updated {
            return Err(AppError::NotConfigured {
                message: format!("no backend named `{id}` is configured"),
            });
        }
        if repointed {
            self.retire_engines_for(id).await;
        }
        self.bus.emit(Change::Backends);
        Ok(())
    }

    /// Soft-remove an external backend (forensic rows keep their target).
    pub(crate) async fn remove_backend(&self, id: &str) -> Result<(), AppError> {
        let _config = self.lock_backend_config(id).await;
        if id == EIDOLA_BACKEND_ID || id == LOCAL_BACKEND_ID {
            return Err(AppError::Config {
                message: format!("backend `{id}` is built in — disable it instead"),
            });
        }
        let conn = self.db_conn().await?;
        if !db::remove_backend(&conn, id, now_ms()).await? {
            return Err(AppError::NotConfigured {
                message: format!("no backend named `{id}` is configured"),
            });
        }
        self.retire_engines_for(id).await;
        self.bus.emit(Change::Backends);
        Ok(())
    }

    /// Stop every engine registered under `backend_id` and forget its engines'
    /// standing reports — the row that gave both their meaning has gone or
    /// changed, and neither may outlive its configuration
    /// (`Inner::retire_backend_engines`).
    ///
    /// Emits [`Change::LocalModels`] when *either* of those happened, since
    /// either changes what the snapshot renders — a backend whose engine had
    /// already failed has no engine to stop and a report to forget, and that
    /// is precisely the case a count of engines could not see. A retirement
    /// that finds nothing stays silent: an invalidation nobody needs would
    /// redraw every subscriber on every disable.
    async fn retire_engines_for(&self, backend_id: &str) {
        // Test seam only: widen the window between the configuration write
        // above and this cleanup. Compiled out of production builds.
        #[cfg(feature = "test-support")]
        {
            let pause = *self
                .backend_config_pause
                .lock()
                .expect("backend config pause");
            if let Some(pause) = pause {
                tokio::time::sleep(pause).await;
            }
        }
        if self.retire_backend_engines(backend_id).changed() {
            self.bus.emit(Change::LocalModels);
        }
    }

    /// The models a backend offers, as selectable entries whose `id` is the
    /// *qualified* selection string ([`qualified_model_id`]).
    ///
    /// - `eidola` → the attested `/models` catalog (honest pricing).
    /// - `openai` → the pinned `model_overrides` when set, else
    ///   `GET /v1/models` with the backend's key. "OpenAI-compatible" does
    ///   not guarantee that endpoint, so a failed fetch surfaces an error
    ///   that suggests pinning models manually.
    /// - `local` / `llamacpp` → the currently loaded engines.
    pub(crate) async fn backend_models(&self, id: &str) -> Result<Vec<ModelInfo>, AppError> {
        let conn = self.db_conn().await?;
        let row = db::get_backend(&conn, id)
            .await?
            .filter(|r| r.removed_at.is_none())
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("no backend named `{id}` is configured"),
            })?;
        let kind = BackendKind::parse(&row.kind).ok_or_else(|| AppError::Database {
            message: format!("unknown backend kind `{}`", row.kind),
        })?;

        match kind {
            BackendKind::Eidola => self.available_models().await,
            BackendKind::OpenAi => {
                if let Some(pinned) = parse_overrides(row.model_overrides.as_deref())? {
                    return Ok(pinned
                        .into_iter()
                        .map(|m| plain_model_info(qualified_model_id(&m, &row.id)))
                        .collect());
                }
                let base_url = row.base_url.as_deref().ok_or_else(|| AppError::Config {
                    message: format!("backend `{id}` has no base URL"),
                })?;
                let client = self.plain_client()?;
                let mut req = client.get(format!("{base_url}/v1/models"));
                if let Some(key) = &row.api_key {
                    req = req.bearer_auth(key);
                }
                let resp = req.send().await.map_err(AppError::from_request)?;
                let status = resp.status();
                let text = resp.text().await.map_err(|e| AppError::Network {
                    // `Response::text` attaches the request URL to its error,
                    // so this goes through the same stripping every other
                    // reqwest error in this crate does.
                    message: format!(
                        "failed to read model list: {}",
                        crate::error::request_error_text(e)
                    ),
                })?;
                if !status.is_success() {
                    return Err(AppError::Server {
                        status: status.as_u16(),
                        message: format!(
                            "model listing failed (HTTP {}) — not every OpenAI-compatible \
                             server offers GET /v1/models; pin this backend's models manually \
                             if yours doesn't",
                            status.as_u16()
                        ),
                    });
                }
                let parsed: OpenAiModelList =
                    serde_json::from_str(&text).map_err(|e| AppError::Network {
                        message: format!("failed to parse model list: {e}"),
                    })?;
                Ok(parsed
                    .data
                    .into_iter()
                    .map(|m| plain_model_info(qualified_model_id(&m.id, &row.id)))
                    .collect())
            }
            BackendKind::Local | BackendKind::LlamaCpp => {
                let state = self.local_models_state().await?;
                let models = if kind == BackendKind::Local {
                    state.models
                } else {
                    state
                        .external
                        .into_iter()
                        .find(|b| b.backend_id == row.id)
                        .map(|b| b.models)
                        .unwrap_or_default()
                };
                // Every on-disk model is selectable — a request against an
                // unloaded one loads its engine on demand. `on_disk` is the
                // whole test, because a snapshot row is not always a file:
                // one still downloading has only a `.part`, and a failed
                // download leaves a row carrying nothing but its error.
                // Offering either would hand the picker an id that can only
                // fail at load time; both stay visible in the model *list*,
                // where the error is the point. Context length is reported
                // for running engines; 0 for not-yet-loaded models (honest:
                // it's not known until the engine starts).
                Ok(models
                    .into_iter()
                    .filter(|m| m.on_disk)
                    .filter_map(|m| match m.status {
                        crate::local_models::LocalModelStatus::Loaded {
                            context_tokens, ..
                        } => Some(ModelInfo {
                            id: m.id,
                            context_length: context_tokens as u64,
                            prompt_credits_per_token: 0.0,
                            completion_credits_per_token: 0.0,
                            request_credits: None,
                        }),
                        crate::local_models::LocalModelStatus::Available
                        | crate::local_models::LocalModelStatus::Loading => Some(ModelInfo {
                            id: m.id,
                            context_length: 0,
                            prompt_credits_per_token: 0.0,
                            completion_credits_per_token: 0.0,
                            request_credits: None,
                        }),
                        crate::local_models::LocalModelStatus::Downloading { .. } => None,
                    })
                    .collect())
            }
        }
    }
}

/// A model whose backend publishes no pricing or context metadata — the
/// honest shape for generic OpenAI-compatible listings.
fn plain_model_info(id: String) -> ModelInfo {
    ModelInfo {
        id,
        context_length: 0,
        prompt_credits_per_token: 0.0,
        completion_credits_per_token: 0.0,
        request_credits: None,
    }
}

/// The subset of `GET /v1/models` both OpenAI and the self-hosted stacks
/// agree on: `data[].id`. Everything else varies too much to rely on.
#[derive(Deserialize)]
struct OpenAiModelList {
    #[serde(default)]
    data: Vec<OpenAiModelEntry>,
}

#[derive(Deserialize)]
struct OpenAiModelEntry {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bare_model_routes_to_eidola() {
        assert_eq!(
            parse_model_ref("gemma4-31b"),
            ModelRef {
                backend_id: "eidola".into(),
                model: "gemma4-31b".into()
            }
        );
    }

    #[test]
    fn parse_local_models_are_qualified_like_any_backend() {
        assert_eq!(
            parse_model_ref("gemma-4-E2B_q4_0-it@local"),
            ModelRef {
                backend_id: "local".into(),
                model: "gemma-4-E2B_q4_0-it".into()
            }
        );
    }

    #[test]
    fn parse_qualified_splits_at_last_at() {
        // Ollama-style tags and HF-style org/model survive the split.
        assert_eq!(
            parse_model_ref("llama3:8b@my-box"),
            ModelRef {
                backend_id: "my-box".into(),
                model: "llama3:8b".into()
            }
        );
        assert_eq!(
            parse_model_ref("org/model@remote"),
            ModelRef {
                backend_id: "remote".into(),
                model: "org/model".into()
            }
        );
    }

    #[test]
    fn parse_degenerate_at_forms_fall_through() {
        // A leading/trailing @ is not a qualification.
        assert_eq!(parse_model_ref("@x").backend_id, "eidola");
        assert_eq!(parse_model_ref("model@").backend_id, "eidola");
    }

    #[test]
    fn qualified_id_round_trips() {
        for (model, backend) in [
            ("gemma4-31b", "eidola"),
            ("tiny", "local"),
            ("qwen3-8b", "my-llama"),
            ("llama3:8b", "ollama-box"),
        ] {
            let q = qualified_model_id(model, backend);
            let parsed = parse_model_ref(&q);
            assert_eq!(parsed.backend_id, backend, "{q}");
            assert_eq!(parsed.model, model, "{q}");
        }
        // The default backend's sugar stays canonical (no redundant
        // qualifier); everything else is spelled out.
        assert_eq!(qualified_model_id("gemma4-31b", "eidola"), "gemma4-31b");
        assert_eq!(qualified_model_id("tiny", "local"), "tiny@local");
    }

    #[test]
    fn backend_id_validation() {
        assert!(validate_backend_id("my-llama").is_ok());
        assert!(validate_backend_id("box2").is_ok());
        assert!(validate_backend_id("").is_err());
        assert!(validate_backend_id("Has-Caps").is_err());
        assert!(validate_backend_id("with space").is_err());
        assert!(validate_backend_id("with@at").is_err());
        assert!(validate_backend_id("eidola").is_err());
        assert!(validate_backend_id("local").is_err());
        assert!(validate_backend_id(&"x".repeat(33)).is_err());
    }

    #[test]
    fn kind_round_trips() {
        for kind in [
            BackendKind::Eidola,
            BackendKind::Local,
            BackendKind::OpenAi,
            BackendKind::LlamaCpp,
        ] {
            assert_eq!(BackendKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(BackendKind::parse("nonsense"), None);
    }
}
