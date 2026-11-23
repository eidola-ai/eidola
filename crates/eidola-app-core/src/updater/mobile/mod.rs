//! The mobile app's self-update: a verified release's app bundle, checked
//! byte for byte and handed to the native shell to stage and activate, once
//! the user approves.
//!
//! ## Pipeline
//!
//! ```text
//!   prepare   → discover_next_release + verify_release, the desktop's own
//!               stages, unchanged: CI's signature bound to the tag, and the
//!               human attestations over that same manifest
//!             → read the bundle row from the verified manifest bytes, and
//!               the container's URL from release.json
//!             → BundleUpdate::Available(PendingBundleUpdate): nothing
//!               downloaded, nothing staged
//!   approve   → (the user's decision; the only way past this point)
//!             → download the container, bounded by what the row lists
//!             → its sha256 is the manifest's, or stop
//!             → unpack it in memory; every file is listed, with the listed
//!               size and sha256, and every listed file is there, or stop
//!             → bundle.begin, bundle.write per file, bundle.seal
//!             → the shell's digest, file count and byte count of what it
//!               stored are the manifest's, or remove it and stop
//!             → bundle.activate: the next document load boots it, on trial
//!   (the app) → location.reload(); the new bundle's boot confirms it
//!               (bundle.confirm), or the shell falls back on the next launch
//! ```
//!
//! **What makes a release real is [`super::verify_release`], and nothing
//! here adds to it or skips it.** This module turns "this manifest is
//! authentic" into "these bytes are the ones the manifest names", which is
//! [`super::install::stage`]'s job on the desktop: the hashes come from the
//! CI-signed manifest, the URL from the unsigned `release.json`, and never a
//! value and the bytes it describes from the same place.
//!
//! **Nothing is downloaded or staged before approval.** [`prepare`] returns
//! a [`PendingBundleUpdate`] holding what the user is shown — the release
//! summary with its attestations, the structural claims comparison, and the
//! digest the shell will show once the bundle runs — and
//! [`PendingBundleUpdate::approve`], which consumes it, is the only path to
//! the shell's store. The interface's approve button calls it, and nothing
//! else does.
//!
//! **Boot health is the shell's and the next bundle's, not the updater's.**
//! An activated bundle boots once on trial; it stays only if it confirms
//! itself (`eidola-mobile`'s boot), and otherwise the shell reverts to the
//! last confirmed bundle on the next launch. [`BundleReport`] reads where
//! that stands, and [`revert_to_embedded`] is the user's own way back.

pub mod container;
pub mod row;
pub mod store;

use sha2::{Digest, Sha256};

use self::row::{BUNDLE_KEY, BundleFile, BundleRow};
pub use self::store::{
    BundleState, BundleStore, BundleStoreError, EMBEDDED, SealedBundle, ServingBundle, StoredBundle,
};
use super::human_attestation::hex_encode;
use super::{
    Discovery, DiscoveryError, Fetcher, ReleaseCheckError, ReleaseEvidence, ReleaseIndex,
    ReleasePins, ReleaseSummary,
};
use crate::updates::{self, ClaimsComparison};

/// What [`prepare`] found.
#[derive(Debug)]
pub enum BundleUpdate {
    /// No release newer than this bundle is marked latest.
    UpToDate { latest_version: Option<String> },
    /// A newer release verified, but its manifest predates the mobile
    /// bundle and records none: a real release this app cannot take.
    Unbundled { summary: Box<ReleaseSummary> },
    /// A newer release verified and records a bundle, ready for the user's
    /// decision.
    Available(Box<PendingBundleUpdate>),
}

/// Find the release to offer next and verify it, exactly as the desktop's
/// update check does ([`super::discover_next_release`],
/// [`super::verify_release`]), then read the bundle it records.
///
/// Downloads only the signed documents; the bundle itself waits for
/// [`PendingBundleUpdate::approve`].
pub async fn prepare(
    fetcher: &Fetcher,
    releases_url: &str,
    installed_version: &str,
    installed_git_commit: Option<&str>,
    pins: &ReleasePins,
) -> Result<BundleUpdate, ReleaseCheckError> {
    let release = match super::discover_next_release(
        fetcher,
        releases_url,
        installed_version,
        installed_git_commit,
    )
    .await
    {
        Ok(Discovery::NothingMarked) => {
            return Ok(BundleUpdate::UpToDate {
                latest_version: None,
            });
        }
        Ok(Discovery::NotNewer { version }) => {
            return Ok(BundleUpdate::UpToDate {
                latest_version: Some(version),
            });
        }
        Ok(Discovery::Next { release, .. }) => release,
        Err(DiscoveryError { error, .. }) => return Err(error),
    };
    let evidence = super::verify_release(fetcher, &release, pins).await?;
    PendingBundleUpdate::from_evidence(&release, evidence)
}

/// A verified release's bundle, waiting for the user.
#[derive(Debug)]
pub struct PendingBundleUpdate {
    summary: ReleaseSummary,
    manifest_sha256: String,
    claims: ClaimsComparison,
    /// The store version the bundle is staged as: the release's tag.
    version: String,
    container_url: String,
    row: BundleRow,
}

impl PendingBundleUpdate {
    /// Read the bundle a verified release records.
    ///
    /// `evidence` is [`super::verify_release`]'s, for `release`; nothing is
    /// read from anywhere else. The manifest must declare a schema this build
    /// accepts (`updates::SUPPORTED_MANIFEST_SCHEMA_VERSIONS`), and is held
    /// to the one it declares: a manifest from before the bundle's schema
    /// records none
    /// ([`BundleUpdate::Unbundled`]), and one from it on records one, well
    /// formed, with a download for it in `release.json`.
    pub fn from_evidence(
        release: &ReleaseIndex,
        evidence: ReleaseEvidence,
    ) -> Result<BundleUpdate, ReleaseCheckError> {
        let refuse = |reason: String| ReleaseCheckError::Rejected(reason);

        // The attestations are bound to `manifest_sha256`; the row is read
        // from `manifest_bytes`. They are one fact, or the row would be read
        // from a manifest nobody attested.
        let actual = hex_encode(&Sha256::digest(&evidence.manifest_bytes));
        if !actual.eq_ignore_ascii_case(&evidence.manifest_sha256) {
            return Err(refuse(format!(
                "the manifest read hashes to {actual}, but the attestations name {}",
                evidence.manifest_sha256
            )));
        }
        if evidence.summary.git_tag != release.git_tag
            || evidence.summary.version != release.version
        {
            return Err(refuse(format!(
                "the evidence is for {} ({}), not for the release {} ({})",
                evidence.summary.version,
                evidence.summary.git_tag,
                release.version,
                release.git_tag
            )));
        }

        let manifest: serde_json::Value = serde_json::from_slice(&evidence.manifest_bytes)
            .map_err(|e| refuse(format!("artifact-manifest.json does not parse: {e}")))?;
        let declared = manifest.get("schema_version").and_then(|v| v.as_u64());
        // The accepted schemas are an allow-list: a manifest in a shape this
        // build was not written for is refused, never read on the chance
        // that the row it looks for kept its meaning. This is the installer,
        // so there is no "treat as update" here; a newer client takes it.
        if !declared.is_some_and(|v| {
            updates::SUPPORTED_MANIFEST_SCHEMA_VERSIONS
                .iter()
                .any(|s| u64::from(*s) == v)
        }) {
            return Err(refuse(format!(
                "artifact-manifest.json declares schema_version {}, which this build does not \
                 accept (it accepts {:?}); install a newer app",
                manifest
                    .get("schema_version")
                    .map_or_else(|| "«missing»".to_string(), |v| v.to_string()),
                updates::SUPPORTED_MANIFEST_SCHEMA_VERSIONS
            )));
        }
        let records_bundle = declared.is_some_and(|v| v >= updates::MOBILE_ARTIFACT_SET_SCHEMA);
        let entry = manifest.get("artifacts").and_then(|a| a.get(BUNDLE_KEY));
        let entry = match (entry, records_bundle) {
            (None, false) => {
                return Ok(BundleUpdate::Unbundled {
                    summary: Box::new(evidence.summary),
                });
            }
            (Some(entry), true) => entry,
            (None, true) => {
                return Err(refuse(format!(
                    "artifact-manifest.json declares schema {} but records no `{BUNDLE_KEY}`, \
                     which every manifest from schema {} on records",
                    declared.unwrap_or_default(),
                    updates::MOBILE_ARTIFACT_SET_SCHEMA
                )));
            }
            (Some(_), false) => {
                return Err(refuse(format!(
                    "artifact-manifest.json records `{BUNDLE_KEY}` under schema {}, which does \
                     not have it; it is recorded from schema {} on",
                    declared.map_or_else(|| "«missing»".into(), |v| v.to_string()),
                    updates::MOBILE_ARTIFACT_SET_SCHEMA
                )));
            }
        };
        let row = BundleRow::from_value(entry)
            .map_err(|e| refuse(format!("artifact-manifest.json's `{BUNDLE_KEY}`: {e}")))?;

        let container_url = release
            .artifacts
            .as_ref()
            .and_then(|artifacts| artifacts.get(BUNDLE_KEY))
            .map(|artifact| artifact.url.clone())
            .ok_or_else(|| {
                refuse(format!(
                    "the manifest records `{BUNDLE_KEY}`, but release.json indexes no download \
                     for it"
                ))
            })?;

        // The release's tag names the staged version: a store version name
        // is the bundle path grammar's one segment, and a tag outside it is
        // a release this app cannot stage under its own name.
        let version = release.git_tag.clone();
        if !is_version_name(&version) {
            return Err(refuse(format!(
                "the release tag `{version}` is not a name the shell's bundle store accepts"
            )));
        }

        let attested = updates::attested_claims(&manifest);
        let expected = updates::expected_claims();
        let deltas = updates::compare_claims(&expected, &attested);

        Ok(BundleUpdate::Available(Box::new(Self {
            summary: evidence.summary,
            manifest_sha256: evidence.manifest_sha256,
            claims: ClaimsComparison {
                expected,
                attested,
                deltas,
            },
            version,
            container_url,
            row,
        })))
    }

    /// What the user approves: the release, and who attested it.
    pub fn summary(&self) -> &ReleaseSummary {
        &self.summary
    }

    /// The sha256 of the manifest every counted attestation names.
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// The manifest's structural claims against this build's expectations,
    /// as the update notification compares them. A non-empty `deltas` is the
    /// desktop's `ClaimsChanged`: authentic, but the threat model moved, and
    /// the interface shows the side-by-side before it offers approval.
    pub fn claims(&self) -> &ClaimsComparison {
        &self.claims
    }

    /// The bundle digest the shell will show in its sheet once this runs,
    /// lowercase hex.
    pub fn bundle_digest(&self) -> &str {
        &self.row.digest
    }

    /// The store version the bundle will be staged as.
    pub fn version_name(&self) -> &str {
        &self.version
    }

    /// The user approved: download the bundle, check every byte against the
    /// manifest, stage it in the shell's store, check what the shell stored,
    /// and activate it for the next document load.
    ///
    /// Nothing reaches the store before every file has been checked. A
    /// failure after staging began removes what was staged. On success the
    /// bundle is active but on trial: the caller reloads, and the new
    /// bundle's boot confirms it.
    pub async fn approve(
        self,
        fetcher: &Fetcher,
        store: &dyn BundleStore,
    ) -> Result<ActivatedBundle, BundleInstallError> {
        let version = self.version.as_str();
        let state = store.state().await?;
        if state.failed.as_deref() == Some(version) {
            return Err(BundleInstallError::PreviouslyFailed {
                version: version.to_string(),
            });
        }
        if state.active == version {
            // Approved before, and activated; the reload has not happened
            // yet, or the trial is under way.
            return Ok(ActivatedBundle {
                version: self.version,
                digest: self.row.digest,
                state,
            });
        }

        // ── the bytes, before anything reaches the shell ────────────────
        let container = fetcher
            .fetch_url_bounded(
                &self.container_url,
                "the app bundle",
                container_bound(&self.row),
            )
            .await
            .map_err(|e| BundleInstallError::Download {
                reason: e.message().to_string(),
            })?;
        let checked = check_container(&self.row, &container)?;

        // ── staging ─────────────────────────────────────────────────────
        // A copy of this version from an earlier attempt that never
        // activated (a staging left by a killed app, or sealed and then not
        // activated) is replaced, not trusted.
        if state.versions.iter().any(|v| v.version == version) {
            store.remove(version).await?;
        }
        store.begin(version).await?;
        let staged = stage_files(store, version, &checked).await;
        let sealed = match staged {
            Ok(()) => store.seal(version).await,
            Err(e) => Err(e),
        };
        let sealed = match sealed {
            Ok(sealed) => sealed,
            Err(e) => {
                let _ = store.remove(version).await;
                return Err(e.into());
            }
        };
        if let Err(e) = check_sealed(&self.row, &sealed) {
            let _ = store.remove(version).await;
            return Err(e);
        }

        let state = store.activate(version).await?;
        Ok(ActivatedBundle {
            version: self.version,
            digest: self.row.digest,
            state,
        })
    }
}

/// A bundle staged, sealed and activated: the next document load boots it on
/// trial.
#[derive(Debug, Clone)]
pub struct ActivatedBundle {
    pub version: String,
    /// The bundle digest, lowercase hex: the shell's own, checked.
    pub digest: String,
    /// The store after activation.
    pub state: BundleState,
}

/// Why an approved update did not activate. Nothing is active that was not
/// before, whichever it is.
#[derive(Debug, thiserror::Error)]
pub enum BundleInstallError {
    /// The container could not be downloaded. Try again later.
    #[error("could not download the app bundle: {reason}")]
    Download { reason: String },
    /// The container is not the one the manifest records.
    #[error(
        "the downloaded app bundle is not the one the release records \
         (expected sha256 {expected}, got {actual})"
    )]
    ContainerHash { expected: String, actual: String },
    /// The container is not one this app reads.
    #[error("the app bundle's container does not read: {0}")]
    Container(String),
    /// A listed file is not in the container.
    #[error("the app bundle has no `{path}`, which the release lists")]
    MissingFile { path: String },
    /// The container holds a file the manifest does not list.
    #[error("the app bundle holds `{path}`, which the release does not list")]
    UnlistedFile { path: String },
    /// A file is not the length the manifest lists.
    #[error("`{path}` is {actual} bytes, but the release lists {expected}")]
    FileSize {
        path: String,
        expected: u64,
        actual: u64,
    },
    /// A file is not the bytes the manifest lists.
    #[error("`{path}` hashes to {actual}, but the release lists {expected}")]
    FileHash {
        path: String,
        expected: String,
        actual: String,
    },
    /// What the shell stored is not what was handed to it.
    #[error("the shell stored a bundle whose {field} is {actual}, not {expected}")]
    Sealed {
        field: &'static str,
        expected: String,
        actual: String,
    },
    /// This version booted on trial once and never confirmed; offering it
    /// again would fail the same way.
    #[error("the app bundle {version} failed to start once already")]
    PreviouslyFailed { version: String },
    #[error(transparent)]
    Store(#[from] BundleStoreError),
}

impl From<BundleInstallError> for crate::error::AppError {
    fn from(e: BundleInstallError) -> Self {
        crate::error::AppError::Update {
            message: e.to_string(),
        }
    }
}

/// The most a container listing these files can be: each file's header and
/// padding, and the end-of-archive record a `tar` writer may pad to.
fn container_bound(row: &BundleRow) -> u64 {
    const BLOCK: u64 = 512;
    const RECORD: u64 = 20 * BLOCK;
    row.files
        .iter()
        .map(|f| BLOCK + f.size.div_ceil(BLOCK) * BLOCK)
        .sum::<u64>()
        + RECORD
}

/// The container's hash is the manifest's, and its files are exactly the
/// listed ones, each of the listed length and hash. Returns each listed file
/// with its bytes, in listing order.
fn check_container<'a>(
    row: &'a BundleRow,
    container: &'a [u8],
) -> Result<Vec<(&'a BundleFile, &'a [u8])>, BundleInstallError> {
    let actual = hex_encode(&Sha256::digest(container));
    if actual != row.container_sha256 {
        return Err(BundleInstallError::ContainerHash {
            expected: row.container_sha256.clone(),
            actual,
        });
    }
    let mut members = container::read_ustar(container).map_err(BundleInstallError::Container)?;
    check_members(row, &mut members)
}

/// Every listed file is a member with the listed length and hash, and no
/// member is left over.
fn check_members<'a>(
    row: &'a BundleRow,
    members: &mut container::Members<'a>,
) -> Result<Vec<(&'a BundleFile, &'a [u8])>, BundleInstallError> {
    let mut checked = Vec::with_capacity(row.files.len());
    for file in &row.files {
        let bytes = members
            .remove(&file.path)
            .ok_or_else(|| BundleInstallError::MissingFile {
                path: file.path.clone(),
            })?;
        if bytes.len() as u64 != file.size {
            return Err(BundleInstallError::FileSize {
                path: file.path.clone(),
                expected: file.size,
                actual: bytes.len() as u64,
            });
        }
        let actual = hex_encode(&Sha256::digest(bytes));
        if actual != file.sha256 {
            return Err(BundleInstallError::FileHash {
                path: file.path.clone(),
                expected: file.sha256.clone(),
                actual,
            });
        }
        checked.push((file, bytes));
    }
    if let Some(path) = members.keys().next() {
        return Err(BundleInstallError::UnlistedFile { path: path.clone() });
    }
    Ok(checked)
}

/// Hand each checked file to the store.
async fn stage_files(
    store: &dyn BundleStore,
    version: &str,
    files: &[(&BundleFile, &[u8])],
) -> Result<(), BundleStoreError> {
    for (file, bytes) in files {
        store.write(version, &file.path, 0, bytes).await?;
    }
    Ok(())
}

/// The shell's account of what it stored agrees with the manifest.
fn check_sealed(row: &BundleRow, sealed: &SealedBundle) -> Result<(), BundleInstallError> {
    let mismatch = |field: &'static str, expected: String, actual: String| {
        Err(BundleInstallError::Sealed {
            field,
            expected,
            actual,
        })
    };
    if !sealed.digest.eq_ignore_ascii_case(&row.digest) {
        return mismatch("digest", row.digest.clone(), sealed.digest.clone());
    }
    if sealed.files != row.files.len() as u64 {
        return mismatch(
            "file count",
            row.files.len().to_string(),
            sealed.files.to_string(),
        );
    }
    if sealed.bytes != row.total_bytes() {
        return mismatch(
            "length",
            row.total_bytes().to_string(),
            sealed.bytes.to_string(),
        );
    }
    Ok(())
}

/// A store version name (`shell-contract.md` → Bundle store): 1 to 64
/// characters of `[A-Za-z0-9._-]`, not starting with `.`.
pub fn is_version_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Where the shell's provisional activation stands, as the interface
/// reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleReport {
    /// The bundle this document was served from.
    pub serving: ServingBundle,
    /// The running bundle is on its trial: it has not confirmed yet, and
    /// will be reverted at the next launch unless it does.
    pub on_trial: bool,
    /// A bundle activated but not yet loaded: the reload that boots it has
    /// not happened.
    pub awaiting_reload: Option<String>,
    /// The last bundle that failed its trial and was reverted.
    pub failed: Option<String>,
}

impl BundleReport {
    pub fn from_state(state: &BundleState) -> Self {
        let awaiting_reload = (state.active != state.serving.version).then(|| state.active.clone());
        Self {
            on_trial: !state.serving.confirmed && state.serving.version == state.active,
            awaiting_reload,
            failed: state.failed.clone(),
            serving: state.serving.clone(),
        }
    }
}

/// Make the bundle built into the shell the one the next document load
/// boots, confirmed: the user's way back from any update, whatever state the
/// store is in.
pub async fn revert_to_embedded(store: &dyn BundleStore) -> Result<BundleState, BundleStoreError> {
    store.activate(EMBEDDED).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_names_are_the_contract_s() {
        for ok in ["v0.0.12", "embedded", "a", &"x".repeat(64)] {
            assert!(is_version_name(ok), "{ok}");
        }
        for bad in ["", ".v1", "v1+build", "v1/2", &"x".repeat(65)] {
            assert!(!is_version_name(bad), "{bad}");
        }
    }

    fn row_of(files: &[(&str, &[u8])], container: &[u8]) -> BundleRow {
        let listed: Vec<BundleFile> = files
            .iter()
            .map(|(path, bytes)| BundleFile {
                path: path.to_string(),
                size: bytes.len() as u64,
                sha256: hex_encode(&Sha256::digest(bytes)),
            })
            .collect();
        BundleRow {
            container_sha256: hex_encode(&Sha256::digest(container)),
            digest: row::bundle_digest(listed.iter().map(|f| (f.path.as_str(), f.sha256.as_str()))),
            files: listed,
        }
    }

    const FILES: &[(&str, &[u8])] = &[("app.wasm", b"\0asm"), ("index.html", b"<html>")];

    #[test]
    fn a_container_bound_covers_what_tar_writes() {
        let container = container::write_ustar(FILES).unwrap();
        let row = row_of(FILES, &container);
        assert!(container.len() as u64 <= container_bound(&row));
        let mut padded = container.clone();
        padded.resize(padded.len().div_ceil(10240) * 10240, 0);
        assert!(padded.len() as u64 <= container_bound(&row));
    }

    #[test]
    fn the_container_is_checked_before_its_members() {
        let container = container::write_ustar(FILES).unwrap();
        let mut row = row_of(FILES, &container);
        row.container_sha256 = "0".repeat(64);
        assert!(matches!(
            check_container(&row, &container),
            Err(BundleInstallError::ContainerHash { .. })
        ));
    }

    #[test]
    fn each_member_is_held_to_its_listing() {
        let container = container::write_ustar(FILES).unwrap();
        let good = row_of(FILES, &container);
        assert_eq!(check_container(&good, &container).unwrap().len(), 2);

        // The listing is the authority, so each disagreement is named by the
        // check that caught it. The container hash matches in every case:
        // these are a manifest that does not describe its own container.
        let mut size = good.clone();
        size.files[0].size += 1;
        assert!(matches!(
            check_container(&size, &container),
            Err(BundleInstallError::FileSize { ref path, .. }) if path == "app.wasm"
        ));

        let mut hash = good.clone();
        hash.files[1].sha256 = "0".repeat(64);
        assert!(matches!(
            check_container(&hash, &container),
            Err(BundleInstallError::FileHash { ref path, .. }) if path == "index.html"
        ));

        let mut missing = good.clone();
        missing.files.push(BundleFile {
            path: "more.js".into(),
            size: 0,
            sha256: "0".repeat(64),
        });
        assert!(matches!(
            check_container(&missing, &container),
            Err(BundleInstallError::MissingFile { ref path }) if path == "more.js"
        ));

        let mut extra = good.clone();
        extra.files.remove(0);
        assert!(matches!(
            check_container(&extra, &container),
            Err(BundleInstallError::UnlistedFile { ref path }) if path == "app.wasm"
        ));
    }

    #[test]
    fn the_shell_s_account_of_the_bundle_is_checked() {
        let container = container::write_ustar(FILES).unwrap();
        let row = row_of(FILES, &container);
        let good = SealedBundle {
            digest: row.digest.clone(),
            files: 2,
            bytes: 10,
        };
        check_sealed(&row, &good).unwrap();
        for (field, sealed) in [
            (
                "digest",
                SealedBundle {
                    digest: "0".repeat(64),
                    ..good.clone()
                },
            ),
            (
                "file count",
                SealedBundle {
                    files: 3,
                    ..good.clone()
                },
            ),
            (
                "length",
                SealedBundle {
                    bytes: 11,
                    ..good.clone()
                },
            ),
        ] {
            assert!(
                matches!(check_sealed(&row, &sealed), Err(BundleInstallError::Sealed { field: f, .. }) if f == field),
                "{field}"
            );
        }
    }

    fn state(
        serving: (&str, bool),
        active: &str,
        confirmed: bool,
        failed: Option<&str>,
    ) -> BundleState {
        BundleState {
            serving: ServingBundle {
                version: serving.0.into(),
                digest: Some("d".into()),
                confirmed: serving.1,
            },
            active: active.into(),
            confirmed,
            previous: EMBEDDED.into(),
            failed: failed.map(Into::into),
            versions: vec![],
        }
    }

    #[test]
    fn the_report_reads_the_provisional_activation() {
        let settled = BundleReport::from_state(&state((EMBEDDED, true), EMBEDDED, true, None));
        assert!(!settled.on_trial && settled.awaiting_reload.is_none());

        let activated = BundleReport::from_state(&state((EMBEDDED, true), "v2", false, None));
        assert_eq!(activated.awaiting_reload.as_deref(), Some("v2"));
        assert!(!activated.on_trial);

        let trial = BundleReport::from_state(&state(("v2", false), "v2", false, None));
        assert!(trial.on_trial && trial.awaiting_reload.is_none());

        let reverted =
            BundleReport::from_state(&state((EMBEDDED, true), EMBEDDED, true, Some("v2")));
        assert_eq!(reverted.failed.as_deref(), Some("v2"));
        assert!(!reverted.on_trial);
    }
}
