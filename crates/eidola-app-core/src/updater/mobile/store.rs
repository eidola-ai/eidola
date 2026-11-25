//! The seam to the shell's bundle store (`mobile/shell-contract.md` →
//! Bundle store): app-core says what to stage and when to activate it, and
//! the embedding app carries each call to its shell. `eidola-mobile`
//! implements it over the contract's `bundle.*` calls, as it implements
//! [`crate::storage::HostStorage`] over `files.*`.
//!
//! The types mirror the contract's results, so the embedding app reads its
//! shell's replies straight into them.

use serde::{Deserialize, Serialize};

use crate::storage::{HostBounds, HostFuture};

/// The name the shell gives the bundle built into it.
pub const EMBEDDED: &str = "embedded";

/// A store call's failure, by the contract's error codes that the updater
/// acts on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BundleStoreError {
    /// `not_found`: no such version.
    #[error("no such bundle version: {0}")]
    NotFound(String),
    /// `invalid_state`: the call is not allowed now (staging a sealed
    /// version, removing the active one, …).
    #[error("the bundle store refused the call now: {0}")]
    InvalidState(String),
    /// Anything else: the bridge, the disk, a reply that does not parse.
    #[error("the bundle store failed: {0}")]
    Failed(String),
}

/// `bundle.seal`'s result: what the shell found on its own disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedBundle {
    /// The bundle digest the shell computed, lowercase hex.
    pub digest: String,
    pub files: u64,
    pub bytes: u64,
}

/// The bundle a document was served from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServingBundle {
    pub version: String,
    /// `None` while the shell is still computing it (the Android shell
    /// answers `null` until it has hashed what it serves).
    #[serde(default)]
    pub digest: Option<String>,
    pub confirmed: bool,
}

/// One stored version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredBundle {
    pub version: String,
    pub sealed: bool,
    /// `None` while staging.
    pub digest: Option<String>,
}

/// `bundle.state`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleState {
    pub serving: ServingBundle,
    /// The version the next document load boots.
    pub active: String,
    /// Whether `active` has been confirmed.
    pub confirmed: bool,
    /// The last confirmed version, which an unconfirmed `active` falls back
    /// to.
    pub previous: String,
    /// The last version that failed its trial.
    pub failed: Option<String>,
    pub versions: Vec<StoredBundle>,
}

/// The shell's bundle store, call for call.
///
/// Every call is applied before its future resolves, and the updater never
/// issues one before the previous resolved (the contract's one queue).
pub trait BundleStore: HostBounds + 'static {
    /// `bundle.begin`: start staging `version`, empty.
    fn begin<'a>(&'a self, version: &'a str) -> HostFuture<'a, Result<(), BundleStoreError>>;
    /// `bundle.write`: `data` at `offset` of a staging version's file. The
    /// implementation splits a write larger than one request may carry.
    fn write<'a>(
        &'a self,
        version: &'a str,
        path: &'a str,
        offset: u64,
        data: &'a [u8],
    ) -> HostFuture<'a, Result<(), BundleStoreError>>;
    /// `bundle.seal`: make it durable, digest it, mark it read-only.
    fn seal<'a>(
        &'a self,
        version: &'a str,
    ) -> HostFuture<'a, Result<SealedBundle, BundleStoreError>>;
    /// `bundle.activate`: boot it next, on trial.
    fn activate<'a>(
        &'a self,
        version: &'a str,
    ) -> HostFuture<'a, Result<BundleState, BundleStoreError>>;
    /// `bundle.remove`.
    fn remove<'a>(&'a self, version: &'a str) -> HostFuture<'a, Result<(), BundleStoreError>>;
    /// `bundle.state`.
    fn state(&self) -> HostFuture<'_, Result<BundleState, BundleStoreError>>;
}
