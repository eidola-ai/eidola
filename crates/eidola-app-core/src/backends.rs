//! The backend registry — *where an ask can be routed*.
//!
//! ## Design
//!
//! A **backend** is a configured inference destination. Four kinds exist
//! today, in decreasing expected frequency of use:
//!
//! - `eidola` — the confidential Eidola service (singleton). Attested
//!   transport, anonymous-credential billing. Its base URL is *not* stored
//!   on the row: the trust-root pin + config override remain the authority.
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
//! A model selection is qualified as `<model>@<backend-id>`, parsed at the
//! **last** `@` (backend ids are validated to `[a-z0-9-]`, so the split is
//! unambiguous even for Ollama-style `name:tag` or HF-style `org/model`
//! ids). Two legacy forms stay first-class: a bare model id routes to
//! `eidola`, and `local/<slug>` routes to `local` — so existing configs,
//! space histories, and CLI invocations keep working unchanged.
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

/// Parse a model selection string. `<model>@<backend>` splits at the last
/// `@`; a bare `local/<slug>` routes to the local singleton (legacy form);
/// anything else routes to `eidola`. Pure and infallible — whether the
/// backend *exists* is the router's question, not the parser's.
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
    if selection.starts_with(crate::local_models::LOCAL_MODEL_PREFIX) {
        return ModelRef {
            backend_id: LOCAL_BACKEND_ID.to_string(),
            model: selection.to_string(),
        };
    }
    ModelRef {
        backend_id: EIDOLA_BACKEND_ID.to_string(),
        model: selection.to_string(),
    }
}

/// The canonical selection string for a (model, backend) pair — the inverse
/// of [`parse_model_ref`]. Eidola models stay bare and `local/<slug>` stays
/// the local singleton's canonical form, so histories and configs written
/// before backends existed remain the canonical spelling.
pub fn qualified_model_id(model: &str, backend_id: &str) -> String {
    match backend_id {
        EIDOLA_BACKEND_ID => model.to_string(),
        LOCAL_BACKEND_ID if model.starts_with(crate::local_models::LOCAL_MODEL_PREFIX) => {
            model.to_string()
        }
        _ => format!("{model}@{backend_id}"),
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
}

// ============================================================================
// Inner methods — the operations `AppCore` wraps.
// ============================================================================

impl Inner {
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
        let conn = self.db_conn().await?;
        if !db::set_backend_enabled(&conn, id, enabled, now_ms()).await? {
            return Err(AppError::NotConfigured {
                message: format!("no backend named `{id}` is configured"),
            });
        }
        self.bus.emit(Change::Backends);
        Ok(())
    }

    /// Update an external backend's configuration. Singletons carry no
    /// editable connection config, so they are refused here.
    pub(crate) async fn update_backend(
        &self,
        id: &str,
        update: BackendUpdate,
    ) -> Result<(), AppError> {
        if id == EIDOLA_BACKEND_ID || id == LOCAL_BACKEND_ID {
            return Err(AppError::Config {
                message: format!("backend `{id}` is built in — only enable/disable applies"),
            });
        }
        let overrides_json = update
            .model_overrides
            .map(|o| overrides_to_json(o.as_deref()));
        let conn = self.db_conn().await?;
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
            now_ms(),
        )
        .await?;
        if !updated {
            return Err(AppError::NotConfigured {
                message: format!("no backend named `{id}` is configured"),
            });
        }
        self.bus.emit(Change::Backends);
        Ok(())
    }

    /// Soft-remove an external backend (forensic rows keep their target).
    pub(crate) async fn remove_backend(&self, id: &str) -> Result<(), AppError> {
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
        self.bus.emit(Change::Backends);
        Ok(())
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
                let client = match &self.http_override {
                    Some(c) => c.clone(),
                    None => crate::local_models::plain_http_client()?,
                };
                let mut req = client.get(format!("{base_url}/v1/models"));
                if let Some(key) = &row.api_key {
                    req = req.bearer_auth(key);
                }
                let resp = req.send().await.map_err(AppError::from_request)?;
                let status = resp.status();
                let text = resp.text().await.map_err(|e| AppError::Network {
                    message: format!("failed to read model list: {e}"),
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
                Ok(models
                    .into_iter()
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
                        _ => None,
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
    fn parse_legacy_local_prefix_routes_to_local() {
        assert_eq!(
            parse_model_ref("local/gemma-4-E2B_q4_0-it"),
            ModelRef {
                backend_id: "local".into(),
                model: "local/gemma-4-E2B_q4_0-it".into()
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
            ("local/tiny", "local"),
            ("qwen3-8b", "my-llama"),
            ("llama3:8b", "ollama-box"),
        ] {
            let q = qualified_model_id(model, backend);
            let parsed = parse_model_ref(&q);
            assert_eq!(parsed.backend_id, backend, "{q}");
            assert_eq!(parsed.model, model, "{q}");
        }
        // The legacy forms stay canonical (no redundant qualifier).
        assert_eq!(qualified_model_id("gemma4-31b", "eidola"), "gemma4-31b");
        assert_eq!(qualified_model_id("local/tiny", "local"), "local/tiny");
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
