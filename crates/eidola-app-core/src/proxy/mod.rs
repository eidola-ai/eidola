//! The **local inference proxy** — an OpenAI-compatible surface other tools on
//! this machine reach Eidola-served models through.
//!
//! This module is the wire contract and the space-free turn behind it. The
//! *listener* that binds the TCP socket lives in the app
//! (`crates/eidola-gui/AGENTS.md` → App lifecycle), because that is the process
//! holding the local database's single-writer lock — exactly the split the
//! local control protocol already takes ([`crate::ipc`]).
//!
//! ## What it is, and what it deliberately is not
//!
//! A request arriving here goes through the **same flow as one the app makes
//! for itself**: the backend registry decides where it routes, an `eidola`
//! route builds an attested client and verifies the enclave on every
//! handshake, a remote call provisions and spends an ACT, and the exchange
//! lands in the Record. What it does *not* do is touch the semantic layer:
//! no space is created, no participant is joined, no action is written, no
//! context is assembled. The downstream tool supplies its own messages; this
//! app has no conversation to add them to and must not invent one.
//!
//! That places it as a **sibling of the chore runner** ([`crate::utility`])
//! rather than a rider on it. The chore runner is the existing precedent for a
//! raw, space-free completion that still pays — and it is explicitly *not* the
//! precedent for evidence: it writes no request rows and records no
//! attestations, because a chore has no action to hang a forensic trail from.
//! The proxy's whole contract is the opposite. A request a tool made through
//! Eidola has to be as visible in the Record as one the app made, or the
//! Record stops being the place a person can go to see what left this machine.
//! So the *resolution* half is shared verbatim
//! ([`crate::Inner::resolve_utility_target`] — cheap, no engine start, no
//! network) and the *opening* half is this module's own, with the attestation
//! observer, the provider and connection rows, and the request row the chore
//! runner deliberately skips. See [`route`].
//!
//! ## Headers: an allowlist, never a forward
//!
//! **The proxy never forwards a downstream header.** It constructs the
//! upstream request itself and carries only what the flow requires, so a
//! header a future tool invents is dropped rather than passed. The rule is
//! stated as an allowlist rather than a denylist for exactly that reason —
//! see [`route::UpstreamHeaders`] for the enumerated set and the reasoning on
//! each member, and [`route::TraceUpstream`] for `traceparent`.
//!
//! Two members are worth naming here because they are the ones a reader will
//! look for:
//!
//! - **`Authorization`** from downstream authenticates the *tool to the
//!   proxy*. It is consumed here and never forwarded. What goes upstream is
//!   the ACT this app spends (or an external backend's own key) — unrelated
//!   credentials that happen to share a header name.
//! - **`traceparent` / `tracestate`** are stripped. The Eidola server treats
//!   an inbound sampled `traceparent` as a request to record that request at
//!   per-request granularity, and generic client-side OpenTelemetry
//!   instrumentation injects one on *every* outbound call — so a tool with
//!   OTel switched on would opt every request out of the aggregate-only
//!   guarantee without anyone deciding to, and carry its own trace id across
//!   requests, which is a self-linking primitive on the anonymous surface.
//!
//! ## Keys
//!
//! Keys are generated here and displayed once. Only a domain-separated
//! SHA-256 of each is stored, so nothing on disk can be replayed at the
//! proxy — see [`generate_key`] and [`key_digest`].

pub mod http;
pub mod route;

use std::net::IpAddr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::changes::Change;
use crate::db;
use crate::error::AppError;
use crate::{AppCore, Inner, join_err, now_ms};

/// The port the proxy binds when nothing has chosen one.
///
/// Deliberately adjacent to the port the on-device-LLM convention has settled
/// on (11434), so a reader recognises the neighbourhood, and deliberately not
/// *equal* to it, so Eidola never fights an Ollama install for a socket.
pub const DEFAULT_BIND_PORT: u16 = 11437;

/// The address the proxy binds when nothing has chosen one.
pub const DEFAULT_BIND_ADDRESS: &str = "127.0.0.1";

/// The leading characters every generated key wears.
///
/// A key is a bearer secret that will end up pasted into other tools' config
/// files and, inevitably, into places it should not be. A recognisable prefix
/// is what lets a person — or a secret scanner — say what a stray string is
/// without having to try it against anything.
pub const KEY_PREFIX: &str = "eid-";

/// How many random bytes back a key. 256 bits, so the digest below is the
/// whole of what protects it: guessing is not a threat model at this size.
const KEY_ENTROPY_BYTES: usize = 32;

/// How much of a key the settings pane may show. Enough to tell two rows
/// apart, far too little to narrow a guess.
const KEY_DISPLAY_PREFIX_LEN: usize = KEY_PREFIX.len() + 6;

/// What an engine-backed backend offers through the proxy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalExposure {
    /// Only models whose engine is already running. Nothing a downstream
    /// request names can start a subprocess or claim memory.
    #[default]
    Loaded,
    /// Every downloaded model, loading its engine on the first request that
    /// names it — the shape a tool expects from a model list, at the cost of a
    /// long first request and whatever the eviction planner has to unload to
    /// make room.
    Downloaded,
}

impl LocalExposure {
    pub fn as_str(self) -> &'static str {
        match self {
            LocalExposure::Loaded => "loaded",
            LocalExposure::Downloaded => "downloaded",
        }
    }

    /// An unrecognized stored value reads as the **narrower** setting, which is
    /// the safe direction: a build that does not understand what was written
    /// must not expose more than the one that wrote it meant to.
    fn parse(s: &str) -> Self {
        match s {
            "downloaded" => LocalExposure::Downloaded,
            _ => LocalExposure::Loaded,
        }
    }
}

/// The proxy's configuration, resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProxySettings {
    /// Whether the proxy should be listening. The listener is the app's to
    /// run; this is the stored intent it reads at launch and on every change.
    pub enabled: bool,
    pub bind_address: String,
    pub bind_port: u16,
    pub local_exposure: LocalExposure,
    /// The backends the proxy exposes, filtered to those that are still live
    /// and enabled — a permission over a backend that is not there offers
    /// nothing.
    pub backends: Vec<String>,
    /// Every backend id an exposure row names, live or not. The settings pane
    /// reads this so a backend the reader disabled still shows its exposure
    /// choice rather than silently losing it.
    pub exposed_ids: Vec<String>,
    /// How many keys are live. The count rather than the keys, because this is
    /// what decides whether the proxy has anyone to let in — a proxy with no
    /// keys refuses everything rather than running open.
    pub live_key_count: i64,
}

impl ProxySettings {
    /// Whether the bind address is loopback.
    ///
    /// **This iteration has no TLS**, so a non-loopback bind puts prompts and
    /// answers on the wire in the clear and offers the proxy to the network.
    /// Nothing refuses it — it is the reader's machine and their call — but
    /// every surface that shows the address says so, and this is the one
    /// predicate all of them read.
    pub fn is_loopback(&self) -> bool {
        self.bind_address
            .parse::<IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
    }
}

/// The fields a settings write may move. Absent means unchanged, the shape
/// every other update struct in this crate takes.
#[derive(Clone, Debug, Default)]
pub struct ProxySettingsUpdate {
    pub enabled: Option<bool>,
    pub bind_address: Option<String>,
    pub bind_port: Option<u16>,
    pub local_exposure: Option<LocalExposure>,
}

/// One key, as the settings pane sees it. The key itself is not here: it
/// existed once, in the answer to [`AppCore::create_proxy_key`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProxyKeyInfo {
    pub id: String,
    pub label: String,
    /// The key's leading characters — enough to tell two rows apart.
    pub prefix: String,
    pub created_at: i64,
    pub last_used_at: Option<i64>,
    pub revoked_at: Option<i64>,
}

impl ProxyKeyInfo {
    pub fn is_live(&self) -> bool {
        self.revoked_at.is_none()
    }
}

/// A key at the one moment it exists in full.
#[derive(Clone, Debug)]
pub struct MintedProxyKey {
    pub info: ProxyKeyInfo,
    /// **Shown once and then unrecoverable.** Only the digest is stored, so
    /// there is no second chance to display this and no way for anything on
    /// disk to reconstruct it.
    pub key: String,
}

/// Generate a key: the prefix, then 256 bits of OS randomness in
/// URL-safe base64 (no padding, so the whole string is one word a shell,
/// a config file and an HTTP header all carry unchanged).
pub fn generate_key() -> String {
    use base64::Engine;
    use rand_core::RngCore;

    let mut bytes = [0u8; KEY_ENTROPY_BYTES];
    rand_core::OsRng.fill_bytes(&mut bytes);
    format!(
        "{KEY_PREFIX}{}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    )
}

/// The stored form of a key: a domain-separated SHA-256, hex.
///
/// **Deliberately a plain hash and not a password KDF.** Argon2 and its
/// siblings exist to make *low-entropy* secrets expensive to guess; this
/// secret is 256 bits of OS randomness that this app generated, so there is
/// nothing to slow down — while a KDF would put tens of milliseconds on the
/// front of every proxied request, on the machine the reader is waiting at.
/// The same argument (and the same construction) is already written down for
/// the account fingerprint in [`crate::config::fingerprint_of`].
///
/// Length-prefixed under its own separator so no other digest in this app can
/// ever collide with one of these by construction.
pub fn key_digest(key: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"eidola.proxy-key.v1\0");
    hasher.update((key.len() as u64).to_be_bytes());
    hasher.update(key.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// The leading characters of a key, for a row that must be identifiable
/// without being usable. Char-boundary safe (the alphabet is ASCII, so this
/// only ever matters if the constant is later changed).
fn display_prefix(key: &str) -> String {
    key.chars().take(KEY_DISPLAY_PREFIX_LEN).collect()
}

/// Validate a bind address as an IP literal.
///
/// A hostname is refused rather than resolved: what a name resolves to is not
/// this app's to decide on a listening socket, and "bind to whatever this name
/// means today" is not a property anyone can reason about.
pub fn parse_bind_address(value: &str) -> Result<IpAddr, AppError> {
    value
        .trim()
        .parse::<IpAddr>()
        .map_err(|_| AppError::Config {
            message: format!(
                "`{}` is not an IP address — the proxy binds an address, not a name",
                value.trim()
            ),
        })
}

impl Inner {
    pub(crate) async fn proxy_settings(&self) -> Result<ProxySettings, AppError> {
        let conn = self.db_conn().await?;
        let stored = db::get_proxy_settings(&conn).await?;
        let backends = db::list_proxy_backends(&conn).await?;
        let exposed_ids = db::list_proxy_backend_rows(&conn).await?;
        let live_key_count = db::live_proxy_key_count(&conn).await?;
        Ok(match stored {
            Some(row) => ProxySettings {
                enabled: row.enabled,
                bind_address: row.bind_address,
                // A port the database cannot represent as a `u16` was never
                // written by this app; falling back beats refusing to answer.
                bind_port: u16::try_from(row.bind_port).unwrap_or(DEFAULT_BIND_PORT),
                local_exposure: LocalExposure::parse(&row.local_exposure),
                backends,
                exposed_ids,
                live_key_count,
            },
            None => ProxySettings {
                enabled: false,
                bind_address: DEFAULT_BIND_ADDRESS.to_string(),
                bind_port: DEFAULT_BIND_PORT,
                local_exposure: LocalExposure::default(),
                backends,
                exposed_ids,
                live_key_count,
            },
        })
    }

    pub(crate) async fn update_proxy_settings(
        &self,
        update: ProxySettingsUpdate,
    ) -> Result<ProxySettings, AppError> {
        // Validate before the first write: a refusal leaves zero trace.
        if let Some(address) = update.bind_address.as_deref() {
            parse_bind_address(address)?;
        }
        if update.bind_port == Some(0) {
            return Err(AppError::Config {
                message: "the proxy needs a port to bind; 0 would take whatever was free, \
                          which is not an address a tool can be told about"
                    .into(),
            });
        }

        // **No read-modify-write.** Each field moves its own column or none
        // at all, so two controls used before the first settles cannot restore
        // each other's old values — see [`db::update_proxy_settings`].
        let bind_address = update.bind_address.map(|a| a.trim().to_string());
        let local_exposure = update.local_exposure.map(|e| e.as_str().to_string());
        let conn = self.db_conn().await?;
        db::update_proxy_settings(
            &conn,
            update.enabled,
            bind_address.as_deref(),
            update.bind_port.map(i64::from),
            local_exposure.as_deref(),
            now_ms(),
        )
        .await?;
        drop(conn);
        self.bus.emit(Change::Proxy);
        self.proxy_settings().await
    }

    pub(crate) async fn set_proxy_backend_exposed(
        &self,
        backend_id: &str,
        exposed: bool,
    ) -> Result<ProxySettings, AppError> {
        let conn = self.db_conn().await?;
        // Named before it is written, so exposing something that does not
        // exist is a refusal rather than a row nothing will ever join.
        if exposed
            && db::get_backend(&conn, backend_id)
                .await?
                .filter(|b| b.removed_at.is_none())
                .is_none()
        {
            return Err(AppError::NotConfigured {
                message: format!("no backend named `{backend_id}` is configured"),
            });
        }
        db::set_proxy_backend(&conn, backend_id, exposed, now_ms()).await?;
        drop(conn);
        self.bus.emit(Change::Proxy);
        self.proxy_settings().await
    }

    pub(crate) async fn proxy_keys(&self) -> Result<Vec<ProxyKeyInfo>, AppError> {
        let conn = self.db_conn().await?;
        Ok(db::list_proxy_keys(&conn)
            .await?
            .into_iter()
            .map(|row| ProxyKeyInfo {
                id: row.id,
                label: row.label,
                prefix: row.prefix,
                created_at: row.created_at,
                last_used_at: row.last_used_at,
                revoked_at: row.revoked_at,
            })
            .collect())
    }

    pub(crate) async fn create_proxy_key(&self, label: String) -> Result<MintedProxyKey, AppError> {
        let label = label.trim().to_string();
        if label.is_empty() {
            return Err(AppError::Config {
                message: "a key needs a name, so a reader can tell later which tool holds it"
                    .into(),
            });
        }
        let key = generate_key();
        let id = Uuid::now_v7().to_string();
        let prefix = display_prefix(&key);
        let now = now_ms();
        let conn = self.db_conn().await?;
        db::insert_proxy_key(&conn, &id, &label, &prefix, &key_digest(&key), now).await?;
        drop(conn);
        self.bus.emit(Change::Proxy);
        Ok(MintedProxyKey {
            info: ProxyKeyInfo {
                id,
                label,
                prefix,
                created_at: now,
                last_used_at: None,
                revoked_at: None,
            },
            key,
        })
    }

    pub(crate) async fn revoke_proxy_key(&self, id: &str) -> Result<bool, AppError> {
        let conn = self.db_conn().await?;
        let revoked = db::revoke_proxy_key(&conn, id, now_ms()).await?;
        drop(conn);
        if revoked {
            self.bus.emit(Change::Proxy);
        }
        Ok(revoked)
    }

    /// Authenticate a presented key, and stamp its last use.
    ///
    /// The digest is what is compared, so a presented secret never reaches the
    /// database and never appears in a query. The touch is best-effort: a
    /// write failure must not cost a request that has already authenticated.
    ///
    /// **The stamp announces itself exactly once per key.** A durable write
    /// that no surface hears about leaves the pane saying "never used"
    /// indefinitely — the store loads its listing at launch and nothing else
    /// invalidates it. But emitting per authentication would put a `Change` on
    /// the bus for every proxied request, and each one costs every window two
    /// reads and a listener reconcile. The pane renders "used" or "never used"
    /// and nothing finer, so the only moment the rendered value moves is the
    /// **first** use: that is when the invalidation is emitted, and the
    /// prior value comes back with the lookup, so asking costs no second query.
    pub(crate) async fn authenticate_proxy_key(&self, presented: &str) -> Result<bool, AppError> {
        let conn = self.db_conn().await?;
        let Some((id, last_used_at)) =
            db::find_live_proxy_key(&conn, &key_digest(presented)).await?
        else {
            return Ok(false);
        };
        let stamped = db::touch_proxy_key(&conn, &id, now_ms()).await.is_ok();
        if stamped && last_used_at.is_none() {
            self.bus.emit(Change::Proxy);
        }
        Ok(true)
    }
}

impl AppCore {
    /// The proxy's resolved configuration. Refresh on [`Change::Proxy`].
    pub async fn proxy_settings(&self) -> Result<ProxySettings, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.proxy_settings().await })
            .await
            .map_err(join_err)?
    }

    /// Move part of the proxy's configuration; answers the resolved settings
    /// so a caller reads what landed rather than what it asked for.
    pub async fn update_proxy_settings(
        &self,
        update: ProxySettingsUpdate,
    ) -> Result<ProxySettings, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.update_proxy_settings(update).await })
            .await
            .map_err(join_err)?
    }

    /// Expose or withdraw one backend.
    pub async fn set_proxy_backend_exposed(
        &self,
        backend_id: String,
        exposed: bool,
    ) -> Result<ProxySettings, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.set_proxy_backend_exposed(&backend_id, exposed).await })
            .await
            .map_err(join_err)?
    }

    /// Every key ever generated, newest first.
    pub async fn proxy_keys(&self) -> Result<Vec<ProxyKeyInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.proxy_keys().await })
            .await
            .map_err(join_err)?
    }

    /// Generate a key. **The secret in the answer is the only copy** — only a
    /// digest is stored.
    pub async fn create_proxy_key(&self, label: String) -> Result<MintedProxyKey, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.create_proxy_key(label).await })
            .await
            .map_err(join_err)?
    }

    /// Revoke a key. Answers whether a live row was actually revoked.
    pub async fn revoke_proxy_key(&self, id: String) -> Result<bool, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.revoke_proxy_key(&id).await })
            .await
            .map_err(join_err)?
    }

    /// Whether a presented key authenticates. Stamps the key's last use.
    pub async fn authenticate_proxy_key(&self, presented: String) -> Result<bool, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.authenticate_proxy_key(&presented).await })
            .await
            .map_err(join_err)?
    }

    /// The models the proxy exposes.
    pub async fn proxy_models(&self) -> Result<Vec<crate::ModelInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.proxy_models().await })
            .await
            .map_err(join_err)?
    }

    /// One non-streaming proxied completion — a **space-free** turn: it pays
    /// and it records, and it writes nothing to the semantic layer.
    pub async fn proxy_chat(
        &self,
        request: route::ProxyChatRequest,
    ) -> Result<route::ProxyChatResponse, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.proxy_chat(request).await })
            .await
            .map_err(join_err)?
    }

    /// One streaming proxied completion. Events reach `sender`; the returned
    /// `Result` is the terminal outcome.
    ///
    /// **The turn outlives the caller that asked for it.** It runs on this
    /// core's own runtime, so a dropped await detaches rather than cancels:
    /// the tokens are spent the moment the request goes upstream, so
    /// discarding the answer would bill for nothing. What a vanished caller
    /// loses is delivery, never the work.
    pub async fn proxy_chat_stream(
        &self,
        request: route::ProxyChatRequest,
        sender: tokio::sync::mpsc::Sender<route::ProxyStreamEvent>,
    ) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.proxy_chat_stream(request, sender).await })
            .await
            .map_err(join_err)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_generated_key_is_prefixed_and_never_repeats() {
        let a = generate_key();
        let b = generate_key();
        assert!(a.starts_with(KEY_PREFIX), "got {a}");
        assert_ne!(a, b, "two keys from 256 bits of OS randomness");
        // The whole key is one URL-safe word: no padding, no separators a
        // config file or a header would have to quote.
        assert!(
            a.trim_start_matches(KEY_PREFIX)
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "got {a}"
        );
    }

    #[test]
    fn the_digest_is_one_way_and_domain_separated() {
        let key = generate_key();
        let digest = key_digest(&key);
        assert_eq!(digest.len(), 64, "sha-256, hex");
        assert!(!digest.contains(&key), "the key is not in its own digest");
        assert_eq!(digest, key_digest(&key), "stable");
        assert_ne!(
            digest,
            crate::config::fingerprint_of(&key, ""),
            "a different digest in this app cannot collide with a proxy key's"
        );
    }

    #[test]
    fn a_display_prefix_identifies_without_narrowing() {
        let key = generate_key();
        let prefix = display_prefix(&key);
        assert_eq!(prefix.len(), KEY_DISPLAY_PREFIX_LEN);
        assert!(key.starts_with(&prefix));
        assert!(
            prefix.len() < key.len() / 2,
            "the shown part is a name, not a head start"
        );
    }

    #[test]
    fn an_unknown_stored_exposure_reads_as_the_narrower_one() {
        assert_eq!(LocalExposure::parse("loaded"), LocalExposure::Loaded);
        assert_eq!(
            LocalExposure::parse("downloaded"),
            LocalExposure::Downloaded
        );
        assert_eq!(
            LocalExposure::parse("everything-a-later-build-invents"),
            LocalExposure::Loaded,
            "a build that does not understand what was written must not expose more"
        );
    }

    #[test]
    fn a_bind_address_is_an_address_and_not_a_name() {
        assert!(parse_bind_address("127.0.0.1").is_ok());
        assert!(parse_bind_address(" ::1 ").is_ok());
        assert!(parse_bind_address("0.0.0.0").is_ok());
        assert!(
            parse_bind_address("localhost").is_err(),
            "a name is refused rather than resolved"
        );
    }

    #[test]
    fn loopback_is_asked_of_the_address_that_will_be_bound() {
        let settings = |address: &str| ProxySettings {
            enabled: true,
            bind_address: address.to_string(),
            bind_port: DEFAULT_BIND_PORT,
            local_exposure: LocalExposure::Loaded,
            backends: Vec::new(),
            exposed_ids: Vec::new(),
            live_key_count: 0,
        };
        assert!(settings("127.0.0.1").is_loopback());
        assert!(settings("::1").is_loopback());
        assert!(!settings("0.0.0.0").is_loopback());
        assert!(!settings("192.168.1.10").is_loopback());
        assert!(
            !settings("not-an-address").is_loopback(),
            "an unparseable address is not evidence of loopback"
        );
    }
}
