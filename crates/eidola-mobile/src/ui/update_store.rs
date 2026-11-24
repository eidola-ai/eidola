//! The self-update, as the interface drives it: the desktop's verified
//! update check (`crates/eidola-gui/src/updates.rs` → `UpdatesDisplay`), and
//! after it the one thing the phone adds — the user's approval, which stages
//! the bundle and activates it (`crate::update`, app-core's
//! `updater::mobile`).
//!
//! **Nothing is staged without the approve tap.** A check holds the verified
//! release as a [`PendingBundleUpdate`] in this store, where nothing but
//! [`UpdateStore::approve`] can reach it, and `approve` is called only by the
//! approve button. Approval consumes it: a second tap finds nothing.
//!
//! **A check and an install exclude each other**: an approval is refused
//! while a check is out (its answer could replace what was approved), a check
//! is refused while an install runs, and an install is never replaced: once
//! one starts, nothing else touches the offer until it settles.
//!
//! **State holds the value, render chooses the words**: every outcome here is
//! typed (the release, the claims, the error), and the screen words it.

use std::rc::Rc;

use eidola_app_core::AppCore;
use eidola_app_core::error::AppError;
use eidola_app_core::updater::mobile::{
    BundleReport, BundleState, BundleUpdate, EMBEDDED, PendingBundleUpdate,
};
use eidola_app_core::updater::{ReleaseCheckError, ReleaseSummary};
use eidola_app_core::updates::ClaimsComparison;
use gpui::{Context, Task};

use super::host::Host;

/// What the update check found, and what became of an approval.
#[derive(Debug, Clone)]
pub enum UpdateDisplay {
    /// No check has run in this session.
    NotChecked,
    /// A check is in flight and nothing else is to be shown.
    Checking,
    UpToDate {
        latest_version: Option<String>,
    },
    /// A newer release verified, but it records no app bundle: a real
    /// release this app cannot take.
    Unbundled {
        version: String,
    },
    /// A newer release verified and records a bundle: waiting for the user.
    Available(Rc<Offer>),
    /// The bytes arrived and did not verify: possibly a fake release or a
    /// compromised channel. Never offered.
    Unverifiable {
        reason: String,
    },
    /// The check could not reach the release feed. Not a security signal.
    CheckFailed {
        reason: String,
    },
    /// Approved: downloading, checking and staging.
    Installing(Rc<Offer>),
    /// Staged and activated; the app restarts into it.
    Activated {
        version: String,
    },
    /// The approved update was refused before it was activated (a byte that
    /// did not match the manifest, the shell's store failing). Nothing was
    /// activated.
    InstallFailed {
        offer: Rc<Offer>,
        error: AppError,
    },
}

/// What the user is shown before approving: the release, who attested it,
/// the claims comparison, and the digest the shell's sheet will show.
#[derive(Debug)]
pub struct Offer {
    pub summary: ReleaseSummary,
    pub manifest_sha256: String,
    pub claims: ClaimsComparison,
    pub bundle_digest: String,
    pub version_name: String,
}

impl Offer {
    fn of(pending: &PendingBundleUpdate) -> Self {
        Self {
            summary: pending.summary().clone(),
            manifest_sha256: pending.manifest_sha256().to_owned(),
            claims: pending.claims().clone(),
            bundle_digest: pending.bundle_digest().to_owned(),
            version_name: pending.version_name().to_owned(),
        }
    }

    /// Its claims differ from this build's expectations: authentic, but the
    /// threat model moved. Default not trusted.
    pub fn claims_changed(&self) -> bool {
        !self.claims.deltas.is_empty()
    }
}

pub struct UpdateStore {
    core: Rc<AppCore>,
    host: Rc<dyn Host>,
    display: UpdateDisplay,
    /// The verified release the display offers; only [`Self::approve`]
    /// takes it.
    pending: Option<PendingBundleUpdate>,
    /// The reader accepted a release whose claims changed (by manifest hash).
    claims_accepted: Option<String>,
    check_task: Option<Task<()>>,
    install_task: Option<Task<()>>,
    /// Where the shell's provisional activation stands.
    report: Option<Result<BundleReport, String>>,
    state: Option<BundleState>,
    report_task: Option<Task<()>>,
    revert_task: Option<Task<()>>,
    revert_error: Option<AppError>,
}

impl UpdateStore {
    pub fn new(core: Rc<AppCore>, host: Rc<dyn Host>, cx: &mut Context<Self>) -> Self {
        let mut store = Self {
            core,
            host,
            display: UpdateDisplay::NotChecked,
            pending: None,
            claims_accepted: None,
            check_task: None,
            install_task: None,
            report: None,
            state: None,
            report_task: None,
            revert_task: None,
            revert_error: None,
        };
        store.refresh_report(cx);
        store
    }

    pub fn display(&self) -> &UpdateDisplay {
        &self.display
    }

    pub fn checking(&self) -> bool {
        self.check_task.is_some()
    }

    pub fn installing(&self) -> bool {
        self.install_task.is_some()
    }

    /// Whether the approve button may act now.
    pub fn can_approve(&self) -> bool {
        self.install_task.is_none() && self.check_task.is_none()
    }

    pub fn report(&self) -> Option<&Result<BundleReport, String>> {
        self.report.as_ref()
    }

    pub fn bundle_state(&self) -> Option<&BundleState> {
        self.state.as_ref()
    }

    pub fn reverting(&self) -> bool {
        self.revert_task.is_some()
    }

    pub fn revert_error(&self) -> Option<&AppError> {
        self.revert_error.as_ref()
    }

    /// Whether the embedded bundle is not what the next load boots, or not
    /// what is running: there is something to go back from.
    pub fn can_revert(&self) -> bool {
        self.state
            .as_ref()
            .is_some_and(|s| s.active != EMBEDDED || s.serving.version != EMBEDDED)
    }

    /// The reader's explicit choice to treat a release whose claims changed
    /// as an update (the desktop's "Treat as Update").
    pub fn claims_accepted_for(&self, offer: &Offer) -> bool {
        self.claims_accepted.as_deref() == Some(offer.manifest_sha256.as_str())
    }

    pub fn accept_claims(&mut self, cx: &mut Context<Self>) {
        if let UpdateDisplay::Available(offer) = &self.display {
            self.claims_accepted = Some(offer.manifest_sha256.clone());
            cx.notify();
        }
    }

    /// Re-read the shell's bundle state.
    pub fn refresh_report(&mut self, cx: &mut Context<Self>) {
        let state = self.host.bundle_state();
        self.report_task = Some(cx.spawn(async move |this, cx| {
            let result = state.await;
            this.update(cx, |this, cx| {
                this.report_task = None;
                this.report = Some(
                    result
                        .as_ref()
                        .map(BundleReport::from_state)
                        .map_err(Clone::clone),
                );
                if let Ok(state) = result {
                    this.state = Some(state);
                }
                cx.notify();
            })
            .ok();
        }));
    }

    /// Run a check now. A re-check over a standing result keeps it on screen;
    /// one already running is not started again. Never while installing.
    pub fn check_now(&mut self, cx: &mut Context<Self>) {
        if self.check_task.is_some() || self.install_task.is_some() {
            return;
        }
        if matches!(
            self.display,
            UpdateDisplay::NotChecked | UpdateDisplay::Activated { .. }
        ) {
            self.display = UpdateDisplay::Checking;
        }
        let check = self.host.check_update(self.core.clone());
        self.check_task = Some(cx.spawn(async move |this, cx| {
            let result = check.await;
            this.update(cx, |this, cx| {
                this.check_task = None;
                this.land_check(result);
                cx.notify();
            })
            .ok();
        }));
        cx.notify();
    }

    fn land_check(&mut self, result: Result<BundleUpdate, ReleaseCheckError>) {
        // Never over an install (none can start while a check is out; this
        // holds the line should that ever change).
        if self.install_task.is_some() {
            return;
        }
        self.pending = None;
        self.display = match result {
            Ok(BundleUpdate::UpToDate { latest_version }) => {
                UpdateDisplay::UpToDate { latest_version }
            }
            Ok(BundleUpdate::Unbundled { summary }) => UpdateDisplay::Unbundled {
                version: summary.version,
            },
            Ok(BundleUpdate::Available(pending)) => {
                let offer = Rc::new(Offer::of(&pending));
                self.pending = Some(*pending);
                UpdateDisplay::Available(offer)
            }
            Err(ReleaseCheckError::Rejected(reason)) => UpdateDisplay::Unverifiable { reason },
            Err(ReleaseCheckError::Unreachable(reason)) => UpdateDisplay::CheckFailed { reason },
        };
    }

    /// **The user approved.** Called only by the approve button: download the
    /// bundle, check every byte against the manifest, stage it in the shell
    /// and activate it; then restart into it. A release whose claims changed
    /// is approved only after the reader accepted the change.
    pub fn approve(&mut self, cx: &mut Context<Self>) {
        // An install runs, or a check is out whose answer would replace the
        // offer being approved: nothing to do until it settles.
        if self.install_task.is_some() || self.check_task.is_some() {
            return;
        }
        let UpdateDisplay::Available(offer) = &self.display else {
            return;
        };
        if offer.claims_changed() && !self.claims_accepted_for(offer) {
            return;
        }
        let Some(pending) = self.pending.take() else {
            return;
        };
        let offer = offer.clone();
        self.display = UpdateDisplay::Installing(offer.clone());
        let install = self.host.install(self.core.clone(), pending);
        let host = self.host.clone();
        self.install_task = Some(cx.spawn(async move |this, cx| {
            let result = install.await;
            this.update(cx, |this, cx| {
                this.install_task = None;
                match result {
                    Ok(activated) => {
                        log::info!(
                            "update {} activated (digest {}); restarting into it",
                            activated.version,
                            activated.digest
                        );
                        this.state = Some(activated.state);
                        this.display = UpdateDisplay::Activated {
                            version: activated.version,
                        };
                        cx.notify();
                        // The next document load boots it on trial; its own
                        // first frame confirms it.
                        host.reload();
                    }
                    Err(error) => {
                        this.display = UpdateDisplay::InstallFailed { offer, error };
                        cx.notify();
                    }
                }
            })
            .ok();
        }));
        cx.notify();
    }

    /// Go back to the bundle built into the app, and restart into it.
    pub fn revert(&mut self, cx: &mut Context<Self>) {
        if self.revert_task.is_some() || self.install_task.is_some() {
            return;
        }
        self.revert_error = None;
        let revert = self.host.revert();
        let host = self.host.clone();
        self.revert_task = Some(cx.spawn(async move |this, cx| {
            let result = revert.await;
            this.update(cx, |this, cx| {
                this.revert_task = None;
                match result {
                    Ok(state) => {
                        this.state = Some(state);
                        cx.notify();
                        host.reload();
                    }
                    Err(e) => {
                        this.revert_error = Some(e);
                        cx.notify();
                    }
                }
            })
            .ok();
        }));
        cx.notify();
    }
}
