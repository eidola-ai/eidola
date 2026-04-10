//! TDX attestation verification using the `dcap-qvl` crate.
//!
//! Quote V4 parsing and signature/TCB verification are delegated to
//! [`dcap_qvl`]. We carry our own Intel PCS collateral fetcher (see
//! [`fetch_and_build_entry`]) rather than enabling dcap-qvl's `report`
//! feature, which pulls in `reqwest` hardcoded to `rustls-tls` and
//! transitively forces `ring` into the workspace — breaking our
//! deterministic Nix release pipeline. Verification itself still goes
//! through `dcap_qvl::verify::rustcrypto::verify`. This module is a thin
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
use dcap_qvl::quote::Quote;
use dcap_qvl::tcb_info::TcbStatus;
use dcap_qvl::verify::rustcrypto::verify as dcap_verify;
use der::Decode as DerDecode;
use serde::Deserialize;
use x509_cert::Certificate;
use x509_cert::ext::pkix::CrlDistributionPoints;
use x509_cert::ext::pkix::name::{DistributionPointName, GeneralName};

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

/// TDX TCB status, mirroring `dcap_qvl::tcb_info::TcbStatus`.
///
/// We re-define the enum here so the public surface of `tinfoil-verifier`
/// does not leak its dependency on dcap-qvl, and so adding a new variant
/// upstream forces a compile-time decision in our `From` impl rather than
/// silently mapping it to a default. The `Display` representation matches
/// dcap-qvl exactly so log lines remain stable across the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TdxTcbStatus {
    UpToDate,
    SWHardeningNeeded,
    ConfigurationNeeded,
    ConfigurationAndSWHardeningNeeded,
    OutOfDate,
    OutOfDateConfigurationNeeded,
    Revoked,
}

impl TdxTcbStatus {
    /// Stable lowercase identifier suitable for use as a metric label.
    pub fn as_metric_label(&self) -> &'static str {
        match self {
            Self::UpToDate => "up_to_date",
            Self::SWHardeningNeeded => "sw_hardening_needed",
            Self::ConfigurationNeeded => "configuration_needed",
            Self::ConfigurationAndSWHardeningNeeded => "configuration_and_sw_hardening_needed",
            Self::OutOfDate => "out_of_date",
            Self::OutOfDateConfigurationNeeded => "out_of_date_configuration_needed",
            Self::Revoked => "revoked",
        }
    }
}

impl std::fmt::Display for TdxTcbStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::UpToDate => "UpToDate",
            Self::SWHardeningNeeded => "SWHardeningNeeded",
            Self::ConfigurationNeeded => "ConfigurationNeeded",
            Self::ConfigurationAndSWHardeningNeeded => "ConfigurationAndSWHardeningNeeded",
            Self::OutOfDate => "OutOfDate",
            Self::OutOfDateConfigurationNeeded => "OutOfDateConfigurationNeeded",
            Self::Revoked => "Revoked",
        })
    }
}

impl From<TcbStatus> for TdxTcbStatus {
    fn from(s: TcbStatus) -> Self {
        match s {
            TcbStatus::UpToDate => Self::UpToDate,
            TcbStatus::SWHardeningNeeded => Self::SWHardeningNeeded,
            TcbStatus::ConfigurationNeeded => Self::ConfigurationNeeded,
            TcbStatus::ConfigurationAndSWHardeningNeeded => Self::ConfigurationAndSWHardeningNeeded,
            TcbStatus::OutOfDate => Self::OutOfDate,
            TcbStatus::OutOfDateConfigurationNeeded => Self::OutOfDateConfigurationNeeded,
            TcbStatus::Revoked => Self::Revoked,
        }
    }
}

/// Merged platform + QE TCB status surfaced after a successful TDX
/// signature verification, before [`TcbPolicy`] is applied.
///
/// Consumers receive this via the optional observer callback on
/// [`crate::AttestingClientConfig`] and can use it to drive metrics or
/// alerting. The observer fires for *every* attestation that completed
/// signature verification, including those the policy subsequently
/// rejects, so operators have full visibility into the population of
/// observed TCB levels — not just the ones that made it through.
#[derive(Debug, Clone)]
pub struct TdxTcbObservation {
    pub status: TdxTcbStatus,
    pub advisory_ids: Vec<String>,
}

/// Observer callback type. Invoked synchronously inside the connector
/// layer for every TDX attestation that completes signature verification,
/// regardless of policy outcome. Implementations must be cheap and
/// non-blocking — they run on the TLS handshake hot path.
pub type TdxObserver = Arc<dyn Fn(&TdxTcbObservation) + Send + Sync>;

/// Verify a TDX Quote V4 against pre-fetched collateral.
///
/// 1. Verifies the quote's ECDSA signature against Intel's root CA
/// 2. Validates TCB policy, TDX module identity, and the `nextUpdate`
///    freshness window on the supplied collateral
/// 3. Applies the caller's [`TcbPolicy`] to the merged platform + QE TCB
///    status (`dcap_verify` itself only rejects `Revoked`; we layer Intel's
///    recommended policy on top)
/// 4. Extracts RTMR1, RTMR2, and report_data
pub fn verify_quote(
    raw_quote: &[u8],
    collateral: &QuoteCollateralV3,
    policy: &TcbPolicy,
    observer: Option<&TdxObserver>,
) -> Result<TdxVerification, Error> {
    let now_secs = unix_now()?;

    let verified = dcap_verify(raw_quote, collateral, now_secs)
        .map_err(|e| Error::Tdx(format!("TDX quote verification failed: {e}")))?;

    // Merge platform + QE statuses, then convert to our own typed
    // observation. `verified.status` is the same value pre-stringified,
    // but matching on a typed enum is sturdier than string comparison
    // and surfaces new TcbStatus variants at compile time.
    let merged = verified.platform_status.clone().merge(&verified.qe_status);
    let observation = TdxTcbObservation {
        status: merged.status.into(),
        advisory_ids: merged.advisory_ids,
    };

    // Fire the observer *before* evaluating the policy so consumers see
    // the full population of TCB levels, including ones we will reject.
    if let Some(observer) = observer {
        observer(&observation);
    }

    policy.evaluate(&observation)?;

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

/// Acceptance policy for the merged platform + QE TCB status returned by
/// `dcap_verify`.
///
/// `dcap-qvl` only rejects `Revoked` itself; every other status — including
/// `OutOfDate*` — is returned as `Ok(VerifiedReport)` for the caller to
/// evaluate. This type encodes Intel's recommended verifier policy on top
/// of that, with an optional escape hatch for advisories the operator has
/// explicitly reviewed.
///
/// - `UpToDate` is accepted silently.
/// - `SWHardeningNeeded`, `ConfigurationNeeded`, and
///   `ConfigurationAndSWHardeningNeeded` are accepted with a `warn!` that
///   logs the matched advisory IDs (relying parties are expected to apply
///   the referenced mitigations out-of-band).
/// - `OutOfDate` and `OutOfDateConfigurationNeeded` are rejected unless an
///   `advisory_allowlist` is configured *and* every advisory ID on the
///   matched TCB level is contained in it. This encodes explicit
///   "we know about INTEL-SA-XXXXX and have decided it is not exploitable
///   in our threat model" decisions; a *new* advisory will surface as
///   "advisory not allowlisted" and fail closed.
/// - `Revoked` is always rejected.
#[derive(Default, Clone)]
pub struct TcbPolicy {
    advisory_allowlist: Vec<String>,
}

impl TcbPolicy {
    /// Default policy: Intel's recommendations, no advisory tolerated past
    /// `OutOfDate`.
    pub fn intel_recommended() -> Self {
        Self::default()
    }

    /// Intel's recommendations plus an explicit advisory-ID allowlist.
    /// `OutOfDate*` levels whose entire advisory set is contained in
    /// `allowlist` are accepted with a warning instead of rejected.
    pub fn with_advisory_allowlist<I, S>(allowlist: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            advisory_allowlist: allowlist.into_iter().map(Into::into).collect(),
        }
    }

    fn allows_advisory(&self, id: &str) -> bool {
        self.advisory_allowlist.iter().any(|a| a == id)
    }

    fn evaluate(&self, observation: &TdxTcbObservation) -> Result<(), Error> {
        match observation.status {
            TdxTcbStatus::UpToDate => Ok(()),
            TdxTcbStatus::SWHardeningNeeded
            | TdxTcbStatus::ConfigurationNeeded
            | TdxTcbStatus::ConfigurationAndSWHardeningNeeded => {
                tracing::warn!(
                    status = %observation.status,
                    advisory_ids = ?observation.advisory_ids,
                    "TDX TCB level requires operator action; accepting per Intel recommendation",
                );
                Ok(())
            }
            TdxTcbStatus::OutOfDate | TdxTcbStatus::OutOfDateConfigurationNeeded => {
                let unrecognized: Vec<&String> = observation
                    .advisory_ids
                    .iter()
                    .filter(|a| !self.allows_advisory(a))
                    .collect();
                if !observation.advisory_ids.is_empty() && unrecognized.is_empty() {
                    tracing::warn!(
                        status = %observation.status,
                        advisory_ids = ?observation.advisory_ids,
                        "TDX TCB level is OutOfDate but every advisory is allowlisted; \
                         accepting under operator policy",
                    );
                    Ok(())
                } else {
                    Err(Error::TcbPolicy(format!(
                        "TDX TCB status is {} with advisory IDs {:?}; not in allowlist: {:?}",
                        observation.status, observation.advisory_ids, unrecognized,
                    )))
                }
            }
            TdxTcbStatus::Revoked => Err(Error::TcbPolicy(format!(
                "TDX TCB status is Revoked with advisory IDs {:?}",
                observation.advisory_ids,
            ))),
        }
    }
}

/// Per-client cache of TDX collateral fetched from Intel PCS, keyed by
/// `(fmspc, ca)`. See the module-level docs for the cache state machine.
pub struct CollateralCache {
    slots: Mutex<HashMap<CacheKey, Arc<Slot>>>,
    /// TLS root store used to validate Intel PCS' cert when the cache
    /// fetches or refreshes collateral. Intel PCS uses a public WebPKI
    /// cert, so callers populate this with the same root store they use
    /// for the rest of the verifier (`webpki-roots` for the in-enclave
    /// server, `rustls-native-certs` for the CLI / macOS app).
    tls_roots: Arc<rustls::RootCertStore>,
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
    pub fn new(tls_roots: Arc<rustls::RootCertStore>) -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
            tls_roots,
        }
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

        let entry = fetch_and_build_entry(raw_quote, &self.tls_roots).await?;
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
        let tls_roots = self.tls_roots.clone();
        tokio::spawn(async move {
            let Ok(_guard) = slot.fetch_lock.try_lock() else {
                tracing::trace!(
                    fmspc = hex::encode_upper(key.fmspc),
                    ca = key.ca,
                    "TDX collateral background refresh skipped: another fetch in flight",
                );
                return;
            };
            match fetch_and_build_entry(&raw_quote, &tls_roots).await {
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

/// Intel's official PCS (Provisioning Certification Service) base URL.
const INTEL_PCS_URL: &str = "https://api.trustedservices.intel.com";

/// JSON envelope returned by Intel PCS' `tcb` endpoint.
#[derive(Deserialize)]
struct TcbInfoResponse {
    #[serde(rename = "tcbInfo")]
    tcb_info: serde_json::Value,
    signature: String,
}

/// JSON envelope returned by Intel PCS' `qe/identity` endpoint.
#[derive(Deserialize)]
struct QeIdentityResponse {
    #[serde(rename = "enclaveIdentity")]
    enclave_identity: serde_json::Value,
    signature: String,
}

/// Fetch a fresh collateral set for `raw_quote` from Intel PCS, parse it,
/// and stamp it with cache freshness metadata.
///
/// We deliberately do not depend on dcap-qvl's `report` feature for this:
/// see the crate-level note in `Cargo.toml`. The endpoints, header names,
/// and response shapes here mirror dcap-qvl's `collateral.rs` exactly so
/// that the resulting `QuoteCollateralV3` round-trips through
/// `dcap_qvl::verify::rustcrypto::verify` byte-for-byte the same way as
/// the upstream fetcher would have produced.
///
/// We never populate `pck_certificate_chain`: PCK certs are per-chip, and
/// `dcap_verify` prefers `collateral.pck_certificate_chain` over the chain
/// embedded in each individual quote when the field is set. Leaving it
/// `None` forces dcap-qvl to fall back to each quote's own embedded chain,
/// which is exactly what we want for an FMSPC-scoped cache.
///
/// A fresh reqwest client is built per fetch (mirrors `sevsnp_crl`'s
/// pattern). In steady state this is one client per FMSPC per refresh
/// cycle (~hourly), so the allocation is negligible.
async fn fetch_and_build_entry(
    raw_quote: &[u8],
    tls_roots: &Arc<rustls::RootCertStore>,
) -> Result<SlotEntry, Error> {
    // Pull FMSPC and CA out of the quote's embedded PCK chain. The Tinfoil
    // enclave only emits cert_type 5 quotes, so `Quote::raw_cert_chain` —
    // which `fmspc()` and `ca()` both call — succeeds.
    let quote = Quote::parse(raw_quote)
        .map_err(|e| Error::Tdx(format!("failed to parse TDX quote: {e}")))?;
    let fmspc = quote
        .fmspc()
        .map_err(|e| Error::Tdx(format!("failed to extract FMSPC: {e}")))?;
    let ca = quote
        .ca()
        .map_err(|e| Error::Tdx(format!("failed to extract CA type: {e}")))?;
    let fmspc_hex = hex::encode_upper(fmspc);

    let client = build_pcs_client(tls_roots)?;

    let pck_crl_url = format!("{INTEL_PCS_URL}/sgx/certification/v4/pckcrl?ca={ca}&encoding=der");
    let tcb_url = format!("{INTEL_PCS_URL}/tdx/certification/v4/tcb?fmspc={fmspc_hex}");
    let qe_identity_url =
        format!("{INTEL_PCS_URL}/tdx/certification/v4/qe/identity?update=standard");

    // Fan out the three independent endpoint fetches. The root CA CRL
    // can't join this batch — its URL is extracted from the QE identity
    // issuer chain — so it stays sequential after the join.
    //
    // PCK CRL body is DER, the issuer chain comes back in a header. TCB
    // info and QE identity are JSON envelopes (`{tcbInfo|enclaveIdentity,
    // signature}`) with their issuer chains also in headers. Intel has
    // historically used both `SGX-TCB-Info-Issuer-Chain` and
    // `TCB-Info-Issuer-Chain` for the TCB endpoint; accept either.
    let pck_crl_fetch = async {
        let resp = client
            .get(&pck_crl_url)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| Error::Tdx(format!("failed to fetch PCK CRL from {pck_crl_url}: {e}")))?;
        let chain = get_url_decoded_header(&resp, "SGX-PCK-CRL-Issuer-Chain")?;
        let body = resp
            .bytes()
            .await
            .map_err(|e| Error::Tdx(format!("failed to read PCK CRL body: {e}")))?
            .to_vec();
        Ok::<_, Error>((body, chain))
    };

    let tcb_fetch = async {
        let resp = client
            .get(&tcb_url)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| Error::Tdx(format!("failed to fetch TCB info from {tcb_url}: {e}")))?;
        let chain = get_url_decoded_header(&resp, "SGX-TCB-Info-Issuer-Chain")
            .or_else(|_| get_url_decoded_header(&resp, "TCB-Info-Issuer-Chain"))?;
        let body = resp
            .text()
            .await
            .map_err(|e| Error::Tdx(format!("failed to read TCB info body: {e}")))?;
        Ok::<_, Error>((body, chain))
    };

    let qe_identity_fetch = async {
        let resp = client
            .get(&qe_identity_url)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| {
                Error::Tdx(format!(
                    "failed to fetch QE identity from {qe_identity_url}: {e}"
                ))
            })?;
        let chain = get_url_decoded_header(&resp, "SGX-Enclave-Identity-Issuer-Chain")?;
        let body = resp
            .text()
            .await
            .map_err(|e| Error::Tdx(format!("failed to read QE identity body: {e}")))?;
        Ok::<_, Error>((body, chain))
    };

    let (
        (pck_crl, pck_crl_issuer_chain),
        (raw_tcb_info, tcb_info_issuer_chain),
        (raw_qe_identity, qe_identity_issuer_chain),
    ) = tokio::try_join!(pck_crl_fetch, tcb_fetch, qe_identity_fetch)?;

    // Root CA CRL: Intel PCS does not serve `rootcacrl` directly, so we
    // mirror dcap-qvl's PCS code path: extract the CRL distribution point
    // URL from the root cert at the end of the QE identity issuer chain
    // and fetch it from there.
    let root_ca_crl = {
        let root_der = last_cert_in_pem_chain(&qe_identity_issuer_chain)?;
        let crl_url = extract_crl_url(&root_der)?
            .ok_or_else(|| Error::Tdx("root cert has no CRL distribution point".to_string()))?;
        let resp = client
            .get(&crl_url)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| Error::Tdx(format!("failed to fetch root CA CRL from {crl_url}: {e}")))?;
        resp.bytes()
            .await
            .map_err(|e| Error::Tdx(format!("failed to read root CA CRL body: {e}")))?
            .to_vec()
    };

    // Parse the TCB info / QE identity JSON envelopes. The `signature`
    // fields are hex-encoded; the canonical inner JSON object is what
    // dcap-qvl re-serializes for signature verification.
    let tcb_info_resp: TcbInfoResponse = serde_json::from_str(&raw_tcb_info)
        .map_err(|e| Error::Tdx(format!("TCB info is not valid JSON: {e}")))?;
    let tcb_info = tcb_info_resp.tcb_info.to_string();
    let tcb_info_signature = hex::decode(&tcb_info_resp.signature)
        .map_err(|e| Error::Tdx(format!("TCB info signature is not valid hex: {e}")))?;

    let qe_identity_resp: QeIdentityResponse = serde_json::from_str(&raw_qe_identity)
        .map_err(|e| Error::Tdx(format!("QE identity is not valid JSON: {e}")))?;
    let qe_identity = qe_identity_resp.enclave_identity.to_string();
    let qe_identity_signature = hex::decode(&qe_identity_resp.signature)
        .map_err(|e| Error::Tdx(format!("QE identity signature is not valid hex: {e}")))?;

    let collateral = QuoteCollateralV3 {
        pck_crl_issuer_chain,
        root_ca_crl,
        pck_crl,
        tcb_info_issuer_chain,
        tcb_info,
        tcb_info_signature,
        qe_identity_issuer_chain,
        qe_identity,
        qe_identity_signature,
        pck_certificate_chain: None,
    };

    let now = unix_now()?;
    let hard_expiry = compute_hard_expiry(&collateral, now);
    Ok(SlotEntry {
        collateral: Arc::new(collateral),
        fetched_at: now,
        hard_expiry,
    })
}

/// Build a fresh reqwest client wired to the verifier's TLS roots, with
/// timeouts matched to dcap-qvl's old PCS client (3-minute total budget,
/// 10-second connect). Intel PCS is generally fast but occasionally
/// stalls; the budget is generous enough to ride that out without
/// wedging the calling task forever.
fn build_pcs_client(tls_roots: &Arc<rustls::RootCertStore>) -> Result<reqwest::Client, Error> {
    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates((**tls_roots).clone())
        .with_no_client_auth();
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(180))
        .use_preconfigured_tls(tls_config)
        .build()
        .map_err(|e| Error::Tdx(format!("failed to build Intel PCS HTTP client: {e}")))
}

/// Fetch a header by name and percent-decode it. Intel PCS percent-encodes
/// PEM newlines (`%0A`) in the issuer-chain headers; dcap-qvl decodes
/// them before storing the chain, and we have to do the same so the bytes
/// we hand back match what `dcap_verify` expects.
fn get_url_decoded_header(resp: &reqwest::Response, name: &str) -> Result<String, Error> {
    let value = resp
        .headers()
        .get(name)
        .ok_or_else(|| Error::Tdx(format!("Intel PCS response missing {name} header")))?
        .to_str()
        .map_err(|e| Error::Tdx(format!("Intel PCS {name} header is not valid UTF-8: {e}")))?;
    let decoded = urlencoding::decode(value).map_err(|e| {
        Error::Tdx(format!(
            "Intel PCS {name} header is not valid percent-encoding: {e}"
        ))
    })?;
    Ok(decoded.into_owned())
}

/// Return the DER bytes of the last certificate in a PEM chain. Intel
/// PCS issuer-chain headers are leaf-first, so the last entry is the
/// root CA whose CRL distribution point we want.
fn last_cert_in_pem_chain(pem_chain: &str) -> Result<Vec<u8>, Error> {
    let pems = pem::parse_many(pem_chain.as_bytes())
        .map_err(|e| Error::Tdx(format!("failed to parse PEM chain: {e}")))?;
    let last = pems
        .into_iter()
        .rfind(|p| p.tag() == "CERTIFICATE")
        .ok_or_else(|| Error::Tdx("PEM chain contains no certificates".to_string()))?;
    Ok(last.into_contents())
}

/// Extract the first URI from a certificate's CRL Distribution Points
/// extension (OID 2.5.29.31). Returns `Ok(None)` when the extension is
/// absent or contains no URI; returns `Err` only on parse errors.
fn extract_crl_url(cert_der: &[u8]) -> Result<Option<String>, Error> {
    let cert = Certificate::from_der(cert_der)
        .map_err(|e| Error::Tdx(format!("failed to parse root certificate DER: {e}")))?;
    let Some(extensions) = &cert.tbs_certificate.extensions else {
        return Ok(None);
    };
    for ext in extensions.iter() {
        if ext.extn_id.to_string() != "2.5.29.31" {
            continue;
        }
        let crl_dist_points = CrlDistributionPoints::from_der(ext.extn_value.as_bytes())
            .map_err(|e| Error::Tdx(format!("failed to parse CRL distribution points: {e}")))?;
        for dist_point in crl_dist_points.0.iter() {
            let Some(DistributionPointName::FullName(general_names)) =
                &dist_point.distribution_point
            else {
                continue;
            };
            for general_name in general_names.iter() {
                if let GeneralName::UniformResourceIdentifier(uri) = general_name {
                    return Ok(Some(uri.to_string()));
                }
            }
        }
    }
    Ok(None)
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

    fn obs(status: TdxTcbStatus, advisories: &[&str]) -> TdxTcbObservation {
        TdxTcbObservation {
            status,
            advisory_ids: advisories.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn policy_accepts_up_to_date() {
        let policy = TcbPolicy::intel_recommended();
        assert!(policy.evaluate(&obs(TdxTcbStatus::UpToDate, &[])).is_ok());
    }

    #[test]
    fn policy_accepts_warning_levels() {
        let policy = TcbPolicy::intel_recommended();
        for status in [
            TdxTcbStatus::SWHardeningNeeded,
            TdxTcbStatus::ConfigurationNeeded,
            TdxTcbStatus::ConfigurationAndSWHardeningNeeded,
        ] {
            assert!(
                policy.evaluate(&obs(status, &["INTEL-SA-00001"])).is_ok(),
                "expected {status:?} to be accepted",
            );
        }
    }

    #[test]
    fn policy_rejects_out_of_date_without_allowlist() {
        let policy = TcbPolicy::intel_recommended();
        let err = policy
            .evaluate(&obs(TdxTcbStatus::OutOfDate, &["INTEL-SA-00837"]))
            .unwrap_err();
        assert!(matches!(err, Error::TcbPolicy(_)));
    }

    #[test]
    fn policy_rejects_out_of_date_with_no_advisories() {
        // OutOfDate with no advisories cannot be allowlisted: there's
        // nothing for the operator to explicitly review.
        let policy = TcbPolicy::with_advisory_allowlist(["INTEL-SA-00837"]);
        let err = policy
            .evaluate(&obs(TdxTcbStatus::OutOfDate, &[]))
            .unwrap_err();
        assert!(matches!(err, Error::TcbPolicy(_)));
    }

    #[test]
    fn policy_accepts_out_of_date_when_all_advisories_allowlisted() {
        let policy = TcbPolicy::with_advisory_allowlist(["INTEL-SA-00837", "INTEL-SA-00614"]);
        assert!(
            policy
                .evaluate(&obs(TdxTcbStatus::OutOfDate, &["INTEL-SA-00837"]))
                .is_ok()
        );
        assert!(
            policy
                .evaluate(&obs(
                    TdxTcbStatus::OutOfDateConfigurationNeeded,
                    &["INTEL-SA-00837", "INTEL-SA-00614"],
                ))
                .is_ok()
        );
    }

    #[test]
    fn policy_rejects_out_of_date_when_any_advisory_unrecognized() {
        let policy = TcbPolicy::with_advisory_allowlist(["INTEL-SA-00837"]);
        let err = policy
            .evaluate(&obs(
                TdxTcbStatus::OutOfDate,
                &["INTEL-SA-00837", "INTEL-SA-99999"],
            ))
            .unwrap_err();
        let Error::TcbPolicy(msg) = err else {
            panic!("expected TcbPolicy error");
        };
        assert!(msg.contains("INTEL-SA-99999"), "got: {msg}");
    }

    #[test]
    fn policy_always_rejects_revoked() {
        // Even with a maximally permissive allowlist, Revoked is rejected.
        let policy = TcbPolicy::with_advisory_allowlist(["INTEL-SA-00837"]);
        let err = policy
            .evaluate(&obs(TdxTcbStatus::Revoked, &["INTEL-SA-00837"]))
            .unwrap_err();
        assert!(matches!(err, Error::TcbPolicy(_)));
    }

    #[test]
    fn dcap_to_local_status_conversion_is_lossless() {
        // Spot check every variant: if dcap-qvl ever adds a new variant,
        // the From impl in this module will fail to compile.
        for (dcap, local) in [
            (TcbStatus::UpToDate, TdxTcbStatus::UpToDate),
            (
                TcbStatus::SWHardeningNeeded,
                TdxTcbStatus::SWHardeningNeeded,
            ),
            (
                TcbStatus::ConfigurationNeeded,
                TdxTcbStatus::ConfigurationNeeded,
            ),
            (
                TcbStatus::ConfigurationAndSWHardeningNeeded,
                TdxTcbStatus::ConfigurationAndSWHardeningNeeded,
            ),
            (TcbStatus::OutOfDate, TdxTcbStatus::OutOfDate),
            (
                TcbStatus::OutOfDateConfigurationNeeded,
                TdxTcbStatus::OutOfDateConfigurationNeeded,
            ),
            (TcbStatus::Revoked, TdxTcbStatus::Revoked),
        ] {
            assert_eq!(TdxTcbStatus::from(dcap), local);
            // Display string must match dcap-qvl's exactly so log lines
            // remain stable across the boundary.
            assert_eq!(local.to_string(), dcap.to_string());
        }
    }
}
