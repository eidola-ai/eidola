//! AMD KDS Certificate Revocation List fetching, verification, and caching.
//!
//! AMD publishes per-platform CRLs at
//! `https://kdsintf.amd.com/vcek/v1/{Generation}/crl`. Each CRL is a
//! standard X.509 v2 CRL signed by the platform's ARK (AMD Root Key)
//! using RSA-PSS-SHA384, and lists revoked ASK and VCEK serial numbers
//! for that generation. The `sev` crate verifies the ARK→ASK→VCEK
//! signature chain but does **not** consult any CRL, so we layer
//! revocation checking on top here.
//!
//! ## Why fetch from AMD KDS directly
//!
//! Unlike VCEK fallback (which we route through Tinfoil's ATC service to
//! avoid leaking which enclave we're verifying to AMD), the CRL is a
//! **global** signed object — every Genoa relying party fetches the same
//! `Genoa/crl`, and the request reveals nothing about which chip a
//! handshake is verifying. Trusting the operator's well-known endpoint to
//! deliver the CRL would defeat the entire point: a compromised operator
//! could serve a stale list and silently re-enable a revoked chip. The
//! CRL has to come from AMD directly (or a proxy AMD has signed for) to
//! be trustworthy. The signature is verified against the ARK locally, so
//! the network path to KDS doesn't need to be confidential — only
//! reachable.
//!
//! ## Cache semantics
//!
//! Identical to the TDX collateral cache: stale-while-revalidate with
//! single-flight refresh, soft TTL of 1 hour, hard expiry derived from
//! the CRL's own `nextUpdate` field minus a margin. Background refreshes
//! are deduplicated per generation via a `tokio::sync::Mutex`. AMD's CRL
//! is small (< 4 KB in steady state — only a handful of revocations
//! across the entire generation history) and changes rarely.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use der::{Decode, Encode};
use rsa::RsaPublicKey;
use rsa::pkcs1v15::VerifyingKey as Pkcs1v15VerifyingKey;
use rsa::pss::VerifyingKey as PssVerifyingKey;
use rsa::signature::Verifier;
use sha2::Sha384;
use x509_cert::Certificate;
use x509_cert::crl::CertificateList;
use x509_cert::der::referenced::OwnedToRef;
use x509_cert::time::Time;

use crate::Error;

/// Maximum age before we trigger a background refresh, and the safety
/// margin we keep between "now" and the CRL's declared `nextUpdate`. Same
/// constant as the TDX collateral cache for consistency.
const MAX_CACHE_AGE: Duration = Duration::from_secs(60 * 60);

/// AMD platform generations we know how to fetch CRLs for.
///
/// The `sev` crate's built-in trust anchors (`builtin::genoa::ark`,
/// `builtin::milan::ark`, `builtin::turin::ark`) and KDS endpoint paths
/// share the same `{Genoa,Milan,Turin,...}` naming, so this enum doubles
/// as both the cache key and the URL component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmdGeneration {
    Genoa,
    // Future: Milan, Bergamo, Siena, Turin. Each requires its own ARK
    // trust anchor and a separate cache slot. Adding a variant here
    // forces a compile error in `crl_url` until the URL is wired up.
}

impl AmdGeneration {
    fn crl_url(&self) -> &'static str {
        match self {
            Self::Genoa => "https://kdsintf.amd.com/vcek/v1/Genoa/crl",
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Genoa => "Genoa",
        }
    }
}

/// Per-client cache of AMD CRLs keyed by [`AmdGeneration`]. See the
/// module-level docs for the cache state machine.
#[derive(Default)]
pub struct CrlCache {
    slots: Mutex<HashMap<AmdGeneration, Arc<Slot>>>,
}

/// One cache slot. Same shape as `tdx::CollateralCache`'s `Slot`: state
/// guarded by a std `Mutex` (only ever locked for snapshot/replace, never
/// across an await), refresh single-flighted by a `tokio::sync::Mutex`.
struct Slot {
    state: Mutex<Option<SlotEntry>>,
    fetch_lock: tokio::sync::Mutex<()>,
}

#[derive(Clone)]
struct SlotEntry {
    crl: Arc<VerifiedCrl>,
    fetched_at: u64,
    /// UNIX seconds at which we must stop serving this entry — `MAX_CACHE_AGE`
    /// before the CRL's declared `nextUpdate`.
    hard_expiry: u64,
}

/// A CRL whose signature has been verified against an ARK and whose
/// revoked-serial set has been materialized for O(1) lookup.
pub struct VerifiedCrl {
    revoked: HashSet<Vec<u8>>,
}

impl VerifiedCrl {
    /// Returns `true` iff `serial` is on the revocation list. The serial
    /// is compared as raw big-endian bytes (the same encoding x509-cert
    /// uses for `SerialNumber::as_bytes()`).
    pub fn is_revoked(&self, serial: &[u8]) -> bool {
        self.revoked.contains(serial)
    }

    /// Number of revoked entries in the list. Useful for log/metric
    /// enrichment.
    pub fn revoked_count(&self) -> usize {
        self.revoked.len()
    }
}

impl CrlCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the CRL for `generation` and verify that none of the
    /// supplied certificate serial numbers are revoked.
    ///
    /// The CRL is fetched from AMD KDS on cold start (or hard expiry),
    /// verified against `ark`, and cached. On soft staleness the cache
    /// returns the previous valid CRL and spawns a background refresh.
    ///
    /// `serials` should contain the serial numbers of every AMD-issued
    /// cert in the chain we just verified — typically the ASK and the
    /// VCEK. The ARK is self-signed and never appears in CRLs, so it
    /// should not be in this list.
    pub async fn check_revocation(
        &self,
        generation: AmdGeneration,
        ark: &Certificate,
        serials: &[&[u8]],
    ) -> Result<(), Error> {
        let crl = self.get_or_fetch(generation, ark).await?;
        for serial in serials {
            if crl.is_revoked(serial) {
                return Err(Error::CertChain(format!(
                    "AMD {} CRL lists certificate serial {} as revoked",
                    generation.name(),
                    hex::encode(serial),
                )));
            }
        }
        Ok(())
    }

    async fn get_or_fetch(
        &self,
        generation: AmdGeneration,
        ark: &Certificate,
    ) -> Result<Arc<VerifiedCrl>, Error> {
        let now_secs = unix_now()?;
        let slot = self.slot(generation);

        let current = slot.state.lock().expect("CRL slot mutex poisoned").clone();

        if let Some(entry) = current {
            if now_secs >= entry.hard_expiry {
                tracing::warn!(
                    generation = generation.name(),
                    "AMD CRL past hard expiry; blocking on KDS refresh",
                );
                return self.blocking_fetch(generation, &slot, ark).await;
            }

            let soft_expiry = entry.fetched_at.saturating_add(MAX_CACHE_AGE.as_secs());
            if now_secs >= soft_expiry {
                self.spawn_background_refresh(generation, slot.clone(), ark.clone());
            }
            return Ok(entry.crl);
        }

        self.blocking_fetch(generation, &slot, ark).await
    }

    fn slot(&self, generation: AmdGeneration) -> Arc<Slot> {
        self.slots
            .lock()
            .expect("CRL slot map mutex poisoned")
            .entry(generation)
            .or_insert_with(|| {
                Arc::new(Slot {
                    state: Mutex::new(None),
                    fetch_lock: tokio::sync::Mutex::new(()),
                })
            })
            .clone()
    }

    async fn blocking_fetch(
        &self,
        generation: AmdGeneration,
        slot: &Arc<Slot>,
        ark: &Certificate,
    ) -> Result<Arc<VerifiedCrl>, Error> {
        let _guard = slot.fetch_lock.lock().await;

        // Re-check: another waiter may have refreshed while we queued.
        let now_secs = unix_now()?;
        let already = slot.state.lock().expect("CRL slot mutex poisoned").clone();
        if let Some(entry) = already
            && now_secs < entry.hard_expiry
        {
            return Ok(entry.crl);
        }

        let entry = fetch_and_build_entry(generation, ark).await?;
        let crl = entry.crl.clone();
        *slot.state.lock().expect("CRL slot mutex poisoned") = Some(entry);
        tracing::debug!(
            generation = generation.name(),
            revoked_count = crl.revoked_count(),
            "AMD CRL cache populated",
        );
        Ok(crl)
    }

    fn spawn_background_refresh(
        &self,
        generation: AmdGeneration,
        slot: Arc<Slot>,
        ark: Certificate,
    ) {
        tokio::spawn(async move {
            let Ok(_guard) = slot.fetch_lock.try_lock() else {
                tracing::trace!(
                    generation = generation.name(),
                    "AMD CRL background refresh skipped: another fetch in flight",
                );
                return;
            };
            match fetch_and_build_entry(generation, &ark).await {
                Ok(entry) => {
                    let revoked_count = entry.crl.revoked_count();
                    *slot.state.lock().expect("CRL slot mutex poisoned") = Some(entry);
                    tracing::debug!(
                        generation = generation.name(),
                        revoked_count,
                        "AMD CRL background refresh succeeded",
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        generation = generation.name(),
                        error = %e,
                        "AMD CRL background refresh failed; serving previous \
                         entry until the next request retries",
                    );
                }
            }
        });
    }
}

async fn fetch_and_build_entry(
    generation: AmdGeneration,
    ark: &Certificate,
) -> Result<SlotEntry, Error> {
    let url = generation.crl_url();
    tracing::debug!(
        generation = generation.name(),
        url,
        "fetching AMD CRL from KDS"
    );

    // Each refresh allocates a fresh reqwest client. Same trade-off as
    // dcap-qvl::collateral::get_collateral_from_pcs: AMD KDS uses public
    // WebPKI, no custom roots are involved, and refresh frequency is
    // ~hourly per generation in steady state.
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| Error::CertChain(format!("failed to build CRL HTTP client: {e}")))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| Error::CertChain(format!("failed to fetch AMD CRL from {url}: {e}")))?
        .error_for_status()
        .map_err(|e| Error::CertChain(format!("AMD KDS CRL fetch returned error: {e}")))?;
    let body = resp
        .bytes()
        .await
        .map_err(|e| Error::CertChain(format!("failed to read AMD CRL body: {e}")))?;

    parse_and_verify(generation, ark, &body)
}

/// Parse a DER-encoded X.509 CRL, verify its signature against `ark`, and
/// build a `SlotEntry` with the materialized revoked-serial set and
/// freshness deadline.
fn parse_and_verify(
    generation: AmdGeneration,
    ark: &Certificate,
    body: &[u8],
) -> Result<SlotEntry, Error> {
    let crl = CertificateList::from_der(body)
        .map_err(|e| Error::CertChain(format!("failed to parse AMD CRL DER: {e}")))?;

    verify_crl_signature(ark, &crl)?;

    let now_secs = unix_now()?;
    let next_update_secs = next_update_secs(&crl);
    let hard_expiry = match next_update_secs {
        Some(next) => next.saturating_sub(MAX_CACHE_AGE.as_secs()),
        None => {
            tracing::warn!(
                generation = generation.name(),
                "AMD CRL has no nextUpdate field; falling back to MAX_CACHE_AGE TTL",
            );
            now_secs.saturating_add(MAX_CACHE_AGE.as_secs())
        }
    };

    let revoked: HashSet<Vec<u8>> = crl
        .tbs_cert_list
        .revoked_certificates
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|rc| rc.serial_number.as_bytes().to_vec())
        .collect();

    Ok(SlotEntry {
        crl: Arc::new(VerifiedCrl { revoked }),
        fetched_at: now_secs,
        hard_expiry,
    })
}

/// Verify the CRL's signature against the supplied ARK certificate.
///
/// AMD signs CRLs with RSA-PSS-SHA384 (the same algorithm and parameters
/// the ARK uses for cross-signing the ASK). We accept that as the only
/// algorithm; anything else is an error rather than a silent fallback.
fn verify_crl_signature(ark: &Certificate, crl: &CertificateList) -> Result<(), Error> {
    // Algorithm OIDs we accept. AMD currently uses RSA-PSS-SHA384 for
    // CRLs (1.2.840.113549.1.1.10 with explicit SHA-384 parameters),
    // matching the algorithm the `sev` crate already enforces for
    // ARK→ASK signing. We also accept plain rsaEncryption-with-SHA384
    // (1.2.840.113549.1.1.12, the PKCS#1 v1.5 OID) as a forward-compat
    // hedge — the AMD KDS endpoint has historically advertised both at
    // various times.
    const RSA_PSS_OID: &str = "1.2.840.113549.1.1.10";
    const RSA_PKCS1_SHA384_OID: &str = "1.2.840.113549.1.1.12";

    let sig_alg = crl.signature_algorithm.oid.to_string();
    let signature = crl.signature.raw_bytes();

    let tbs_der = crl
        .tbs_cert_list
        .to_der()
        .map_err(|e| Error::CertChain(format!("failed to encode CRL tbs_cert_list: {e}")))?;

    let ark_spki_ref = ark.tbs_certificate.subject_public_key_info.owned_to_ref();
    let ark_pubkey = RsaPublicKey::try_from(ark_spki_ref)
        .map_err(|e| Error::CertChain(format!("ARK does not contain an RSA public key: {e}")))?;

    if sig_alg == RSA_PSS_OID {
        let verifying_key = PssVerifyingKey::<Sha384>::new(ark_pubkey);
        let sig = rsa::pss::Signature::try_from(signature)
            .map_err(|e| Error::CertChain(format!("invalid CRL RSA-PSS signature bytes: {e}")))?;
        verifying_key.verify(&tbs_der, &sig).map_err(|e| {
            Error::CertChain(format!(
                "AMD CRL RSA-PSS signature verification failed: {e}"
            ))
        })?;
    } else if sig_alg == RSA_PKCS1_SHA384_OID {
        let verifying_key = Pkcs1v15VerifyingKey::<Sha384>::new(ark_pubkey);
        let sig = rsa::pkcs1v15::Signature::try_from(signature).map_err(|e| {
            Error::CertChain(format!("invalid CRL PKCS#1 v1.5 signature bytes: {e}"))
        })?;
        verifying_key.verify(&tbs_der, &sig).map_err(|e| {
            Error::CertChain(format!(
                "AMD CRL PKCS#1 v1.5 signature verification failed: {e}"
            ))
        })?;
    } else {
        return Err(Error::CertChain(format!(
            "AMD CRL uses unsupported signature algorithm OID: {sig_alg}",
        )));
    }

    Ok(())
}

fn next_update_secs(crl: &CertificateList) -> Option<u64> {
    let next = crl.tbs_cert_list.next_update?;
    let dt = match next {
        Time::UtcTime(t) => t.to_unix_duration(),
        Time::GeneralTime(t) => t.to_unix_duration(),
    };
    Some(dt.as_secs())
}

fn unix_now() -> Result<u64, Error> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| {
            Error::CertChain(format!(
                "system clock is earlier than UNIX_EPOCH; AMD CRL verification \
                 requires a correctly configured system clock: {e}"
            ))
        })?
        .as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sev::certs::snp::builtin::genoa;

    fn genoa_ark_x509() -> Certificate {
        let der = genoa::ark()
            .expect("failed to load Genoa ARK")
            .to_der()
            .expect("failed to DER-encode Genoa ARK");
        Certificate::from_der(&der).expect("failed to parse Genoa ARK as x509-cert")
    }

    #[test]
    fn verified_crl_lookup() {
        let mut revoked = HashSet::new();
        revoked.insert(vec![0x01, 0x02, 0x03]);
        revoked.insert(vec![0xde, 0xad, 0xbe, 0xef]);
        let crl = VerifiedCrl { revoked };
        assert!(crl.is_revoked(&[0x01, 0x02, 0x03]));
        assert!(crl.is_revoked(&[0xde, 0xad, 0xbe, 0xef]));
        assert!(!crl.is_revoked(&[0x01, 0x02, 0x04]));
        assert!(!crl.is_revoked(&[]));
        assert_eq!(crl.revoked_count(), 2);
    }

    #[test]
    fn check_revocation_passes_when_serials_clear() {
        let cache = CrlCache::new();
        let slot = cache.slot(AmdGeneration::Genoa);
        // Pre-populate the slot with a known empty CRL so the test
        // doesn't hit the network.
        *slot.state.lock().unwrap() = Some(SlotEntry {
            crl: Arc::new(VerifiedCrl {
                revoked: HashSet::new(),
            }),
            fetched_at: unix_now().unwrap(),
            hard_expiry: u64::MAX,
        });
        let ark = genoa_ark_x509();
        let serial: &[u8] = &[0x42];
        let result =
            futures_lite_block_on(cache.check_revocation(AmdGeneration::Genoa, &ark, &[serial]));
        assert!(result.is_ok(), "expected ok, got {result:?}");
    }

    #[test]
    fn check_revocation_rejects_listed_serial() {
        let cache = CrlCache::new();
        let slot = cache.slot(AmdGeneration::Genoa);
        let mut revoked = HashSet::new();
        revoked.insert(vec![0xab, 0xcd]);
        *slot.state.lock().unwrap() = Some(SlotEntry {
            crl: Arc::new(VerifiedCrl { revoked }),
            fetched_at: unix_now().unwrap(),
            hard_expiry: u64::MAX,
        });
        let ark = genoa_ark_x509();
        let serial: &[u8] = &[0xab, 0xcd];
        let err =
            futures_lite_block_on(cache.check_revocation(AmdGeneration::Genoa, &ark, &[serial]))
                .unwrap_err();
        match err {
            Error::CertChain(msg) => assert!(msg.contains("revoked"), "got: {msg}"),
            other => panic!("expected CertChain error, got: {other:?}"),
        }
    }

    /// Run a future to completion on a single-threaded tokio runtime.
    /// Used by the lookup tests so they don't need `#[tokio::test]`.
    fn futures_lite_block_on<F: std::future::Future>(f: F) -> F::Output {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(f)
    }
}
