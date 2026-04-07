//! Structured enclave measurement types.
//!
//! Mirrors the shape used in Tinfoil's `tinfoil-deployment.json` (predicate
//! `snp-tdx-multiplatform/v1`) and our own `artifact-manifest.json`. Tinfoil
//! ships every release on both AMD SEV-SNP and Intel TDX hosts, so a single
//! "release" is identified by a pair of measurements — one per platform.
//!
//! At verification time the runtime platform is detected from the attestation
//! document, and the appropriate field of each [`EnclaveMeasurement`] is
//! checked. The matching value is returned to callers as a [`MatchedMeasurement`]
//! so they can tell which platform actually answered.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Allowed code measurements for a single Tinfoil enclave release, on both
/// supported hardware platforms.
///
/// Field names match the JSON shape used by `tinfoil-deployment.json` and our
/// own `artifact-manifest.json`, so config files and manifests can round-trip
/// through this type via serde.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnclaveMeasurement {
    /// Hex-encoded SEV-SNP launch digest (48 bytes / 96 hex characters).
    pub snp_measurement: String,
    /// Intel TDX runtime measurement registers.
    pub tdx_measurement: TdxMeasurement,
}

/// Intel TDX runtime measurement registers used by Tinfoil's deterministic
/// measurement scheme. RTMR1 binds the kernel/initrd/cmdline; RTMR2 binds the
/// dm-verity-protected rootfs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TdxMeasurement {
    /// Hex-encoded RTMR1 (48 bytes / 96 hex characters).
    pub rtmr1: String,
    /// Hex-encoded RTMR2 (48 bytes / 96 hex characters).
    pub rtmr2: String,
}

/// The measurement that was actually observed and matched during a successful
/// attestation. Only the platform that produced the attestation is populated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatchedMeasurement {
    /// AMD SEV-SNP launch digest, hex-encoded.
    SevSnp(String),
    /// Intel TDX runtime measurements, hex-encoded.
    Tdx(TdxMeasurement),
}

impl fmt::Display for MatchedMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SevSnp(m) => write!(f, "sev-snp:{m}"),
            Self::Tdx(t) => write!(f, "tdx:rtmr1={},rtmr2={}", t.rtmr1, t.rtmr2),
        }
    }
}
