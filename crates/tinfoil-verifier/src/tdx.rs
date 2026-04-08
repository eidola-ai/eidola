//! TDX attestation verification using the `dcap-qvl` crate.
//!
//! Quote V4 parsing, Intel PCS collateral fetching, and signature/TCB
//! verification are all delegated to [`dcap_qvl`]. This module is a thin
//! adapter that maps dcap-qvl's APIs onto our error type and adds a
//! per-client cache of fetched collateral keyed by `(fmspc, ca)`.
//!
//! ## Cache semantics: stale-while-revalidate, single-flight
//!
//! - Fresh entries (`age < MAX_CACHE_AGE`, well clear of `nextUpdate`) are
//!   returned immediately.
//! - Soft-stale entries (older than `MAX_CACHE_AGE` but still within their
//!   `nextUpdate` safety margin) are returned immediately, *and* a
//!   background refresh is kicked off if one is not already running for
//!   that key. While the refresh is in flight (or retrying after a brief
//!   Intel PCS outage) callers keep getting the previous valid collateral,
//!   so we never absorb a sudden latency spike or hard failure on the
//!   handshake hot path.
//! - Hard-expired entries (within `MAX_CACHE_AGE` of `nextUpdate`) cannot
//!   be served — `dcap_verify` would reject them — so callers block on a
//!   fresh fetch.
//!
//! Refreshes (background and blocking alike) are single-flight per key via
//! a per-slot `tokio::sync::Mutex`, so a thundering herd of new TLS
//! handshakes against a previously-unseen FMSPC results in exactly one
//! Intel PCS round trip.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::DateTime;
use dcap_qvl::QuoteCollateralV3;
use dcap_qvl::collateral::get_collateral_from_pcs;
use dcap_qvl::quote::Quote;
use dcap_qvl::verify::rustcrypto::verify as dcap_verify;
use serde::Deserialize;

use crate::Error;

/// Maximum age before we trigger a background refresh of a cache entry, and
/// also the safety margin we keep between "now" and a collateral's declared
/// `nextUpdate` (so we never hand out collateral that is on the brink of
/// being rejected by `dcap_verify`).
///
/// Intel publishes TCB advisories on roughly a monthly cadence, so a 1-hour
/// soft TTL is conservative — it bounds how long a TCB rotation can sit
/// unnoticed inside this process — without making us noisy on Intel PCS.
const MAX_CACHE_AGE: Duration = Duration::from_secs(60 * 60);

/// Result of a successful TDX quote verification.
pub struct TdxVerification {
    /// RTMR1 (48 bytes).
    pub rtmr1: [u8; 48],
    /// RTMR2 (48 bytes).
    pub rtmr2: [u8; 48],
    /// Full report_data (64 bytes). First 32 bytes = TLS fingerprint.
    pub report_data: [u8; 64],
}

/// Verify a TDX Quote V4 against pre-fetched collateral.
///
/// 1. Verifies the quote's ECDSA signature against Intel's root CA
/// 2. Validates TCB policy, TDX module identity, and the `nextUpdate`
///    freshness window on the supplied collateral
/// 3. Extracts RTMR1, RTMR2, and report_data
pub fn verify_quote(
    raw_quote: &[u8],
    collateral: &QuoteCollateralV3,
) -> Result<TdxVerification, Error> {
    let now_secs = unix_now()?;

    let verified = dcap_verify(raw_quote, collateral, now_secs)
        .map_err(|e| Error::Tdx(format!("TDX quote verification failed: {e}")))?;

    let td_report = verified
        .report
        .as_td10()
        .ok_or_else(|| Error::Tdx("expected TDX report, got SGX".to_string()))?;

    Ok(TdxVerification {
        rtmr1: td_report.rt_mr1,
        rtmr2: td_report.rt_mr2,
        report_data: td_report.report_data,
    })
}

/// Per-client cache of TDX collateral fetched from Intel PCS, keyed by
/// `(fmspc, ca)`. See the module-level docs for the cache state machine.
#[derive(Default)]
pub struct CollateralCache {
    slots: Mutex<HashMap<CacheKey, Arc<Slot>>>,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct CacheKey {
    fmspc: [u8; 6],
    ca: &'static str,
}

/// One cache slot. Lives behind an `Arc` so the slot map is decoupled from
/// the per-key state and the per-key fetch lock.
struct Slot {
    /// Current best entry for this key, if any. Guarded by a std `Mutex`;
    /// the lock is never held across an `await`, only long enough to clone
    /// an `Arc<QuoteCollateralV3>` out or replace the entry after a fetch.
    state: Mutex<Option<SlotEntry>>,
    /// Held by whichever task is currently fetching collateral for this
    /// key. Background refreshes use `try_lock` and bail if a refresh is
    /// already in progress; blocking refreshes acquire it normally and
    /// re-check the slot after they get it (so a queue of waiters
    /// collapses to a single fetch).
    fetch_lock: tokio::sync::Mutex<()>,
}

#[derive(Clone)]
struct SlotEntry {
    collateral: Arc<QuoteCollateralV3>,
    /// UNIX seconds when this entry was last successfully fetched.
    fetched_at: u64,
    /// UNIX seconds at which `dcap_verify` is liable to start rejecting
    /// this collateral. Computed as
    /// `min(tcb.nextUpdate, qe.nextUpdate) - MAX_CACHE_AGE`.
    hard_expiry: u64,
}

impl CollateralCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return cached collateral for `raw_quote`, fetching from Intel PCS if
    /// the entry is missing or hard-expired, and kicking off a background
    /// refresh if it is merely soft-stale.
    pub async fn get_or_fetch(&self, raw_quote: &[u8]) -> Result<Arc<QuoteCollateralV3>, Error> {
        let key = quote_cache_key(raw_quote)?;
        let now_secs = unix_now()?;
        let slot = self.slot(key);

        // Snapshot the current entry without holding the std mutex across
        // any await — `Option::clone()` only bumps the inner Arc refcount.
        let current = slot
            .state
            .lock()
            .expect("collateral slot mutex poisoned")
            .clone();

        if let Some(entry) = current {
            if now_secs >= entry.hard_expiry {
                tracing::warn!(
                    fmspc = hex::encode_upper(key.fmspc),
                    ca = key.ca,
                    "TDX collateral past hard expiry; blocking on Intel PCS refresh",
                );
                return self.blocking_fetch(key, &slot, raw_quote).await;
            }

            let soft_expiry = entry.fetched_at.saturating_add(MAX_CACHE_AGE.as_secs());
            if now_secs >= soft_expiry {
                self.spawn_background_refresh(key, slot.clone(), raw_quote.to_vec());
            }
            return Ok(entry.collateral);
        }

        // Cold: nothing usable in the cache.
        self.blocking_fetch(key, &slot, raw_quote).await
    }

    /// Look up (or insert) the slot for a key.
    fn slot(&self, key: CacheKey) -> Arc<Slot> {
        self.slots
            .lock()
            .expect("collateral slot map mutex poisoned")
            .entry(key)
            .or_insert_with(|| {
                Arc::new(Slot {
                    state: Mutex::new(None),
                    fetch_lock: tokio::sync::Mutex::new(()),
                })
            })
            .clone()
    }

    /// Block until a usable entry exists for `key`, or fail.
    ///
    /// Single-flight via `fetch_lock`: queued callers re-check the slot
    /// after acquiring the lock and return the freshly-stored entry
    /// without re-fetching.
    async fn blocking_fetch(
        &self,
        key: CacheKey,
        slot: &Arc<Slot>,
        raw_quote: &[u8],
    ) -> Result<Arc<QuoteCollateralV3>, Error> {
        let _guard = slot.fetch_lock.lock().await;

        // Re-check: another waiter may have just refreshed the slot while
        // we were queued on `fetch_lock`.
        let now_secs = unix_now()?;
        let already = slot
            .state
            .lock()
            .expect("collateral slot mutex poisoned")
            .clone();
        if let Some(entry) = already
            && now_secs < entry.hard_expiry
        {
            return Ok(entry.collateral);
        }

        let entry = fetch_and_build_entry(raw_quote).await?;
        let collateral = entry.collateral.clone();
        *slot.state.lock().expect("collateral slot mutex poisoned") = Some(entry);
        tracing::debug!(
            fmspc = hex::encode_upper(key.fmspc),
            ca = key.ca,
            "TDX collateral cache populated",
        );
        Ok(collateral)
    }

    /// Spawn a background refresh task. No-ops if a refresh (or blocking
    /// fetch) is already in flight for this key. The task logs success and
    /// failure but never propagates errors — the previous entry stays in
    /// place on failure and the next caller will try again.
    fn spawn_background_refresh(&self, key: CacheKey, slot: Arc<Slot>, raw_quote: Vec<u8>) {
        tokio::spawn(async move {
            let Ok(_guard) = slot.fetch_lock.try_lock() else {
                tracing::trace!(
                    fmspc = hex::encode_upper(key.fmspc),
                    ca = key.ca,
                    "TDX collateral background refresh skipped: another fetch in flight",
                );
                return;
            };
            match fetch_and_build_entry(&raw_quote).await {
                Ok(entry) => {
                    *slot.state.lock().expect("collateral slot mutex poisoned") = Some(entry);
                    tracing::debug!(
                        fmspc = hex::encode_upper(key.fmspc),
                        ca = key.ca,
                        "TDX collateral background refresh succeeded",
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        fmspc = hex::encode_upper(key.fmspc),
                        ca = key.ca,
                        error = %e,
                        "TDX collateral background refresh failed; serving previous \
                         entry until the next request retries",
                    );
                }
            }
        });
    }
}

async fn fetch_and_build_entry(raw_quote: &[u8]) -> Result<SlotEntry, Error> {
    // dcap-qvl builds its own reqwest client internally; we accept that
    // cost (one client per refresh, ~hourly per FMSPC in steady state) in
    // exchange for not maintaining a parallel impl of collateral fetching.
    let mut collateral = get_collateral_from_pcs(raw_quote)
        .await
        .map_err(|e| Error::Tdx(format!("failed to fetch collateral from Intel PCS: {e}")))?;

    // Drop the per-CPU PCK certificate chain that `get_collateral_from_pcs`
    // attaches from the *first* quote that populated this slot. PCK certs
    // are per-chip, so reusing one CPU's chain to verify a different CPU's
    // quote on the same FMSPC would make `dcap_verify` reject the QE
    // report signature (it prefers `collateral.pck_certificate_chain` over
    // the chain embedded in the quote when the field is set; see
    // dcap-qvl's `verify_pck_cert_chain`). Setting it to `None` forces
    // dcap-qvl to fall back to each quote's own embedded chain, which is
    // exactly what we want for an FMSPC-scoped cache.
    collateral.pck_certificate_chain = None;

    let now = unix_now()?;
    let hard_expiry = compute_hard_expiry(&collateral, now);
    Ok(SlotEntry {
        collateral: Arc::new(collateral),
        fetched_at: now,
        hard_expiry,
    })
}

fn quote_cache_key(raw_quote: &[u8]) -> Result<CacheKey, Error> {
    let quote = Quote::parse(raw_quote)
        .map_err(|e| Error::Tdx(format!("failed to parse TDX quote: {e}")))?;
    let fmspc = quote
        .fmspc()
        .map_err(|e| Error::Tdx(format!("failed to extract FMSPC: {e}")))?;
    let ca = quote
        .ca()
        .map_err(|e| Error::Tdx(format!("failed to extract CA type: {e}")))?;
    Ok(CacheKey { fmspc, ca })
}

fn unix_now() -> Result<u64, Error> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| {
            Error::Tdx(format!(
                "system clock is earlier than UNIX_EPOCH; TDX quote verification \
                 requires a correctly configured system clock: {e}"
            ))
        })?
        .as_secs())
}

#[derive(Deserialize)]
struct WithNextUpdate {
    #[serde(rename = "nextUpdate")]
    next_update: String,
}

/// Compute the moment we must stop serving an entry — `MAX_CACHE_AGE`
/// before the earlier of `tcb_info.nextUpdate` and `qe_identity.nextUpdate`.
/// Falls back to `now + MAX_CACHE_AGE` (i.e., serve at most one age cycle)
/// when either field is missing or unparseable.
fn compute_hard_expiry(collateral: &QuoteCollateralV3, now_secs: u64) -> u64 {
    let tcb = parse_next_update(&collateral.tcb_info);
    let qe = parse_next_update(&collateral.qe_identity);
    let next_update = match (tcb, qe) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => return now_secs.saturating_add(MAX_CACHE_AGE.as_secs()),
    };
    next_update.saturating_sub(MAX_CACHE_AGE.as_secs())
}

fn parse_next_update(json: &str) -> Option<u64> {
    let parsed: WithNextUpdate = serde_json::from_str(json).ok()?;
    let dt = DateTime::parse_from_rfc3339(&parsed.next_update).ok()?;
    let secs = dt.timestamp();
    if secs < 0 { None } else { Some(secs as u64) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collateral_with(tcb_next: &str, qe_next: &str) -> QuoteCollateralV3 {
        QuoteCollateralV3 {
            pck_crl_issuer_chain: String::new(),
            root_ca_crl: Vec::new(),
            pck_crl: Vec::new(),
            tcb_info_issuer_chain: String::new(),
            tcb_info: format!(r#"{{"nextUpdate":"{tcb_next}"}}"#),
            tcb_info_signature: Vec::new(),
            qe_identity_issuer_chain: String::new(),
            qe_identity: format!(r#"{{"nextUpdate":"{qe_next}"}}"#),
            qe_identity_signature: Vec::new(),
            pck_certificate_chain: None,
        }
    }

    #[test]
    fn hard_expiry_picks_earlier_next_update() {
        // tcb expires later, qe expires earlier — qe should win
        let collateral = collateral_with("2030-01-02T00:00:00Z", "2030-01-01T00:00:00Z");
        let qe_secs = DateTime::parse_from_rfc3339("2030-01-01T00:00:00Z")
            .unwrap()
            .timestamp() as u64;
        let deadline = compute_hard_expiry(&collateral, 0);
        assert_eq!(deadline, qe_secs - MAX_CACHE_AGE.as_secs());
    }

    #[test]
    fn hard_expiry_falls_back_when_unparseable() {
        let collateral = collateral_with("garbage", "also garbage");
        let now = 1_000_000;
        let deadline = compute_hard_expiry(&collateral, now);
        assert_eq!(deadline, now + MAX_CACHE_AGE.as_secs());
    }
}
