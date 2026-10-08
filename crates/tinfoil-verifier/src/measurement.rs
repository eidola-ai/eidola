//! Enclave measurement types: what a release records, and what a caller pins.
//!
//! Two shapes live here and they answer different questions.
//!
//! [`EnclaveMeasurement`] is the *record*: the JSON shape of Tinfoil's
//! `tinfoil-deployment.json` (predicate `snp-tdx-multiplatform/v1`) and of our
//! own `artifact-manifest.json`, pairing a SEV-SNP launch digest with the
//! RTMR1/RTMR2 values the same release would extend on Intel TDX. Recording a
//! TDX value is not accepting it: RTMR1/RTMR2 are guest-extendable, so a pin
//! made from a record is SEV-SNP-only (see [`AllowedMeasurement::from`]).
//!
//! [`AllowedMeasurement`] is the *pin*: one platform-tagged entry of the set a
//! verifier accepts. The verifier dispatches on the platform of the presented
//! evidence and consults only entries of that platform; a platform with no
//! entry is refused before its evidence is even authenticated. A document's
//! author therefore cannot select a platform branch the caller did not pin —
//! a TDX quote against a SEV-SNP-only pin set fails exactly as it did before
//! TDX acceptance existed.

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Error;
use crate::bundle::Platform;

/// Recorded code measurements for a single Tinfoil enclave release, on both
/// hardware platforms.
///
/// Field names match the JSON shape used by `tinfoil-deployment.json` and our
/// own `artifact-manifest.json`, so config files and manifests can round-trip
/// through this type via serde.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnclaveMeasurement {
    /// Hex-encoded SEV-SNP launch digest (48 bytes / 96 hex characters).
    pub snp_measurement: String,
    /// Intel TDX runtime measurement registers, recorded only.
    pub tdx_measurement: TdxMeasurement,
}

/// Intel TDX runtime measurement registers in Tinfoil's kernel-cmdline
/// measurement scheme. RTMR1 binds the kernel/initrd/cmdline; RTMR2 binds the
/// dm-verity-protected rootfs. Recorded for transparency; never accepted on
/// their own, because guest firmware can extend any value into them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TdxMeasurement {
    /// Hex-encoded RTMR1 (48 bytes / 96 hex characters).
    pub rtmr1: String,
    /// Hex-encoded RTMR2 (48 bytes / 96 hex characters).
    pub rtmr2: String,
}

/// One entry of the set of measurements a verifier accepts.
///
/// The platform is part of the entry: an entry pins exactly one platform's
/// launch identity, and evidence from a platform with no entry is refused.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllowedMeasurement {
    /// The platform-specific launch identity this entry accepts.
    pub platform: PlatformMeasurement,
    /// The number of NVIDIA GPUs the enclave must present evidence for.
    ///
    /// When set to `Some(n)`, the attestation's device evidence must carry
    /// exactly `n` `nvidia-gpu-evidence/v1` items, each bound to this
    /// handshake's nonce, and no GPU item in any other format. The evidence
    /// is counted and nonce-checked, not appraised: NVIDIA's signatures and
    /// reference measurements are not verified here. `None` places no
    /// requirement on device evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_gpus: Option<u32>,
}

/// The platform half of an [`AllowedMeasurement`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum PlatformMeasurement {
    /// An AMD SEV-SNP launch digest (lowercase or uppercase hex, 48 bytes).
    SevSnp {
        /// Hex-encoded launch digest.
        measurement: String,
    },
    /// An Intel TDX launch identity and the platform policy it runs under.
    Tdx(TdxPin),
}

/// An Intel TDX pin in the IGVM launch model (Tinfoil CVM v0.15 and later).
///
/// The launch is identified by MRTD, which the TDX module measures over every
/// page of the IGVM image and which is therefore fixed per image release, and
/// by MRCONFIGID, which carries the SHA-256 of the deployment's config
/// followed by sixteen zero bytes and which the guest refuses to boot without.
/// The image extends no runtime register, so RTMR0–RTMR3 must all be zero —
/// a non-zero RTMR means something extended it after launch. Every other
/// report field describes the machine rather than the image and is pinned by
/// [`TdxPolicy`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TdxPin {
    /// Hex-encoded MRTD (48 bytes).
    pub mrtd: String,
    /// Hex-encoded MRCONFIGID (48 bytes). See [`TdxPin::mrconfigid_for_config`].
    pub mrconfigid: String,
    /// The platform policy the quote must satisfy.
    pub policy: TdxPolicy,
}

impl TdxPin {
    /// The MRCONFIGID a guest launched with `config` reports: the SHA-256 of
    /// the exact config bytes followed by sixteen zero bytes, hex-encoded.
    pub fn mrconfigid_for_config(config: &[u8]) -> String {
        let mut field = [0u8; 48];
        field[..32].copy_from_slice(&Sha256::digest(config));
        hex::encode(field)
    }
}

/// Machine-side Intel TDX policy: the report fields an image cannot fix.
///
/// Every member is required. Values are lowercase or uppercase hex of the
/// exact field width.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TdxPolicy {
    /// Allowed MR_SEAM values (the TDX module's measurement, 48 bytes each).
    /// Must be non-empty.
    pub mr_seam: Vec<String>,
    /// Exact TD_ATTRIBUTES (8 bytes). Must have DEBUG clear and
    /// SEPT_VE_DISABLE set; MRTD does not cover these bits, so a debug
    /// relaunch of the pinned image is caught only here.
    pub td_attributes: String,
    /// Exact XFAM (8 bytes).
    pub xfam: String,
    /// Component-wise minimum TEE_TCB_SVN (16 bytes).
    pub minimum_tee_tcb_svn: String,
    /// Minimum `tcbEvaluationDataNumber` of the Intel TCB Info and QE
    /// Identity collateral the quote was verified against.
    pub minimum_tcb_evaluation_data_number: u32,
    /// Exact QE vendor id from the quote header (16 bytes).
    pub qe_vendor_id: String,
    /// Allowed platform FMSPCs (6 bytes each), taken from the Intel-signed
    /// PCK certificate. `None` accepts any FMSPC Intel's collateral covers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fmspc: Option<Vec<String>>,
}

impl AllowedMeasurement {
    /// A SEV-SNP pin for the given hex launch digest.
    pub fn sev_snp(measurement: impl Into<String>) -> Self {
        Self {
            platform: PlatformMeasurement::SevSnp {
                measurement: measurement.into(),
            },
            expected_gpus: None,
        }
    }

    /// An Intel TDX pin.
    pub fn tdx(pin: TdxPin) -> Self {
        Self {
            platform: PlatformMeasurement::Tdx(pin),
            expected_gpus: None,
        }
    }

    /// Require exactly `gpus` nonce-bound NVIDIA GPU evidence items.
    pub fn with_expected_gpus(mut self, gpus: u32) -> Self {
        self.expected_gpus = Some(gpus);
        self
    }

    /// The platform this entry pins.
    pub fn platform(&self) -> Platform {
        match self.platform {
            PlatformMeasurement::SevSnp { .. } => Platform::SevSnp,
            PlatformMeasurement::Tdx(_) => Platform::Tdx,
        }
    }
}

/// A release record pins its SEV-SNP launch digest only. The recorded TDX
/// RTMR1/RTMR2 are dropped on purpose: they are guest-extendable, so a pin
/// built from them would let any firmware on a genuine TDX machine pass.
impl From<&EnclaveMeasurement> for AllowedMeasurement {
    fn from(record: &EnclaveMeasurement) -> Self {
        Self::sev_snp(record.snp_measurement.clone())
    }
}

/// The launch identity of a TDX quote that matched a pin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TdxLaunchMeasurement {
    /// Hex-encoded MRTD.
    pub mrtd: String,
    /// Hex-encoded MRCONFIGID.
    pub mrconfigid: String,
}

/// The measurement that was actually observed and matched during a successful
/// attestation. Only the platform that produced the attestation is populated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatchedMeasurement {
    /// AMD SEV-SNP launch digest, hex-encoded.
    SevSnp(String),
    /// Intel TDX launch identity, hex-encoded.
    Tdx(TdxLaunchMeasurement),
}

impl fmt::Display for MatchedMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SevSnp(m) => write!(f, "sev-snp:{m}"),
            Self::Tdx(t) => write!(f, "tdx:mrtd={},mrconfigid={}", t.mrtd, t.mrconfigid),
        }
    }
}

/// A validated, decoded [`AllowedMeasurement`]. Built once at client
/// construction so a malformed pin fails there rather than per handshake.
#[derive(Clone, Debug)]
pub(crate) struct CompiledPin {
    pub platform: CompiledPlatform,
    pub expected_gpus: Option<u32>,
}

#[derive(Clone, Debug)]
pub(crate) enum CompiledPlatform {
    SevSnp { measurement: [u8; 48] },
    Tdx(Box<CompiledTdxPin>),
}

impl CompiledPin {
    pub fn platform(&self) -> Platform {
        match self.platform {
            CompiledPlatform::SevSnp { .. } => Platform::SevSnp,
            CompiledPlatform::Tdx(_) => Platform::Tdx,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompiledTdxPin {
    pub mrtd: [u8; 48],
    pub mrconfigid: [u8; 48],
    pub mr_seam: Vec<[u8; 48]>,
    pub td_attributes: [u8; 8],
    pub xfam: [u8; 8],
    pub minimum_tee_tcb_svn: [u8; 16],
    pub minimum_tcb_evaluation_data_number: u32,
    pub qe_vendor_id: [u8; 16],
    pub fmspc: Option<Vec<[u8; 6]>>,
}

/// TD_ATTRIBUTES bit 0: the TD is debuggable by the host.
const TD_ATTRIBUTES_DEBUG: u64 = 1 << 0;
/// TD_ATTRIBUTES bit 28: EPT violations are not converted to #VE.
const TD_ATTRIBUTES_SEPT_VE_DISABLE: u64 = 1 << 28;

pub(crate) fn compile_pins(pins: &[AllowedMeasurement]) -> Result<Vec<CompiledPin>, Error> {
    pins.iter()
        .enumerate()
        .map(|(index, pin)| {
            let platform = match &pin.platform {
                PlatformMeasurement::SevSnp { measurement } => CompiledPlatform::SevSnp {
                    measurement: pin_hex(measurement, index, "sev_snp.measurement")?,
                },
                PlatformMeasurement::Tdx(tdx) => {
                    CompiledPlatform::Tdx(Box::new(compile_tdx(tdx, index)?))
                }
            };
            Ok(CompiledPin {
                platform,
                expected_gpus: pin.expected_gpus,
            })
        })
        .collect()
}

fn compile_tdx(pin: &TdxPin, index: usize) -> Result<CompiledTdxPin, Error> {
    let policy = &pin.policy;
    if policy.mr_seam.is_empty() {
        return Err(Error::InvalidPin(format!(
            "allowed measurement {index}: tdx.policy.mr_seam must list at least one TDX module"
        )));
    }
    let mr_seam = policy
        .mr_seam
        .iter()
        .map(|seam| pin_hex(seam, index, "tdx.policy.mr_seam"))
        .collect::<Result<Vec<_>, _>>()?;
    let td_attributes: [u8; 8] = pin_hex(&policy.td_attributes, index, "tdx.policy.td_attributes")?;
    let attributes = u64::from_le_bytes(td_attributes);
    if attributes & TD_ATTRIBUTES_DEBUG != 0 || attributes & TD_ATTRIBUTES_SEPT_VE_DISABLE == 0 {
        return Err(Error::InvalidPin(format!(
            "allowed measurement {index}: tdx.policy.td_attributes must have DEBUG clear and \
             SEPT_VE_DISABLE set"
        )));
    }
    let fmspc = match &policy.fmspc {
        None => None,
        Some(list) if list.is_empty() => {
            return Err(Error::InvalidPin(format!(
                "allowed measurement {index}: tdx.policy.fmspc, when present, must not be empty"
            )));
        }
        Some(list) => Some(
            list.iter()
                .map(|f| pin_hex(f, index, "tdx.policy.fmspc"))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };
    Ok(CompiledTdxPin {
        mrtd: pin_hex(&pin.mrtd, index, "tdx.mrtd")?,
        mrconfigid: pin_hex(&pin.mrconfigid, index, "tdx.mrconfigid")?,
        mr_seam,
        td_attributes,
        xfam: pin_hex(&policy.xfam, index, "tdx.policy.xfam")?,
        minimum_tee_tcb_svn: pin_hex(
            &policy.minimum_tee_tcb_svn,
            index,
            "tdx.policy.minimum_tee_tcb_svn",
        )?,
        minimum_tcb_evaluation_data_number: policy.minimum_tcb_evaluation_data_number,
        qe_vendor_id: pin_hex(&policy.qe_vendor_id, index, "tdx.policy.qe_vendor_id")?,
        fmspc,
    })
}

fn pin_hex<const N: usize>(value: &str, index: usize, field: &str) -> Result<[u8; N], Error> {
    let bytes = hex::decode(value).map_err(|e| {
        Error::InvalidPin(format!(
            "allowed measurement {index}: {field} is not hex: {e}"
        ))
    })?;
    bytes.try_into().map_err(|_| {
        Error::InvalidPin(format!(
            "allowed measurement {index}: {field} must be exactly {N} bytes"
        ))
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn tdx_pin() -> TdxPin {
        TdxPin {
            mrtd: "11".repeat(48),
            mrconfigid: TdxPin::mrconfigid_for_config(b"config"),
            policy: TdxPolicy {
                mr_seam: vec!["22".repeat(48)],
                td_attributes: "0000001000000000".to_string(),
                xfam: "e702060000000000".to_string(),
                minimum_tee_tcb_svn: "03010200000000000000000000000000".to_string(),
                minimum_tcb_evaluation_data_number: 17,
                qe_vendor_id: "939a7233f79c4ca9940a0db3957f0607".to_string(),
                fmspc: None,
            },
        }
    }

    #[test]
    fn mrconfigid_is_config_sha256_with_sixteen_zero_bytes() {
        let id = TdxPin::mrconfigid_for_config(b"abc");
        assert_eq!(
            id,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\
             00000000000000000000000000000000"
        );
    }

    #[test]
    fn record_conversion_pins_sev_snp_only() {
        let record = EnclaveMeasurement {
            snp_measurement: "ab".repeat(48),
            tdx_measurement: TdxMeasurement {
                rtmr1: "cd".repeat(48),
                rtmr2: "ef".repeat(48),
            },
        };
        let pin = AllowedMeasurement::from(&record);
        assert_eq!(pin.platform(), Platform::SevSnp);
        assert_eq!(pin, AllowedMeasurement::sev_snp("ab".repeat(48)));
    }

    #[test]
    fn compile_accepts_a_well_formed_tdx_pin() {
        let compiled = compile_pins(&[AllowedMeasurement::tdx(tdx_pin())]).unwrap();
        let CompiledPlatform::Tdx(tdx) = &compiled[0].platform else {
            panic!("expected a TDX pin");
        };
        assert_eq!(tdx.mrtd, [0x11; 48]);
        assert_eq!(tdx.td_attributes, [0, 0, 0, 0x10, 0, 0, 0, 0]);
    }

    #[test]
    fn compile_rejects_malformed_and_unsafe_tdx_pins() {
        type Mutation = Box<dyn Fn(&mut TdxPin)>;
        let cases: Vec<(&str, Mutation)> = vec![
            ("tdx.mrtd", Box::new(|p| p.mrtd = "11".repeat(47))),
            (
                "tdx.mrconfigid",
                Box::new(|p| p.mrconfigid = "zz".repeat(48)),
            ),
            ("mr_seam must list", Box::new(|p| p.policy.mr_seam.clear())),
            (
                "DEBUG clear",
                Box::new(|p| p.policy.td_attributes = "0100001000000000".to_string()),
            ),
            (
                "SEPT_VE_DISABLE set",
                Box::new(|p| p.policy.td_attributes = "0000000000000000".to_string()),
            ),
            (
                "tdx.policy.xfam",
                Box::new(|p| p.policy.xfam = "e7".to_string()),
            ),
            (
                "fmspc, when present",
                Box::new(|p| p.policy.fmspc = Some(vec![])),
            ),
            (
                "tdx.policy.fmspc",
                Box::new(|p| p.policy.fmspc = Some(vec!["00".repeat(5)])),
            ),
        ];
        for (needle, mutate) in cases {
            let mut pin = tdx_pin();
            mutate(&mut pin);
            let err = compile_pins(&[AllowedMeasurement::tdx(pin)])
                .expect_err(needle)
                .to_string();
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }

    #[test]
    fn pins_round_trip_through_strict_json() {
        let pin = AllowedMeasurement::tdx(tdx_pin()).with_expected_gpus(8);
        let json = serde_json::to_value(&pin).unwrap();
        assert_eq!(json["expected_gpus"], 8);
        assert!(json["platform"]["tdx"]["policy"]["mr_seam"].is_array());
        let back: AllowedMeasurement = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(back, pin);

        let mut unknown = json;
        unknown["platform"]["tdx"]["rtmr1"] = serde_json::Value::String("00".repeat(48));
        assert!(serde_json::from_value::<AllowedMeasurement>(unknown).is_err());
    }
}
