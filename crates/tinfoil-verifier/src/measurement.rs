//! Enclave pins: the launch identities a caller accepts.
//!
//! [`AllowedMeasurement`] is one platform-tagged entry of the set a
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
    /// An AMD SEV-SNP launch identity.
    SevSnp(SevSnpPin),
    /// An Intel TDX launch identity and the platform policy it runs under.
    Tdx(TdxPin),
}

/// An AMD SEV-SNP pin in the IGVM launch model (Tinfoil CVM v0.15 and later).
///
/// The launch digest is a function of the release's IGVM image alone (every
/// page the host adds, then the boot processor's VMSA; hosts launch with
/// `KVM_SEV_SNP_MEASURE_BSP_ONLY`, so the processor count is not in it, and
/// the RAM size reaches the guest through an unmeasured parameter page), so
/// it is the same for every deployment of that image. What distinguishes a
/// deployment is HOST_DATA, which the host passes at launch, the firmware
/// copies into every report, and the guest checks against its config before
/// booting: the SHA-256 of the deployment's config. The guest policy is
/// chosen by the host at launch and is not in the digest either: a relaunch
/// of the pinned image with DEBUG set keeps the same digest while handing
/// the hypervisor `SNP_DBG_DECRYPT`, so the policy is pinned too. All three
/// are required; a pin without HOST_DATA would accept every deployment of
/// the image, anyone's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SevSnpPin {
    /// Hex-encoded launch digest (48 bytes).
    pub measurement: String,
    /// Hex-encoded HOST_DATA (32 bytes). See [`SevSnpPin::host_data_for_config`].
    pub host_data: String,
    /// The guest policy the image is built for, spelled `0x` + lowercase
    /// hex with no leading zeros (`0x30133`). A report's policy must equal it
    /// in every bit except the firmware ABI minimum (bits 0-15), which may be
    /// higher (a launch asking for newer firmware). The pin must have DEBUG
    /// and MIGRATE_MA clear and the must-be-one bit 17 set.
    pub guest_policy: String,
}

impl SevSnpPin {
    /// The HOST_DATA a guest launched with `config` reports: the SHA-256 of
    /// the exact config bytes, hex-encoded.
    pub fn host_data_for_config(config: &[u8]) -> String {
        hex::encode(Sha256::digest(config))
    }
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
    /// The machines allowed, by PPID (16 bytes each): the Platform Provisioning
    /// ID in the Intel-signed PCK certificate's SGX extension, which names one
    /// registered platform. Must be non-empty, so every TDX pin names the
    /// individual hosts it accepts.
    pub ppids: Vec<String>,
    /// Expected `DynamicPlatform` flag of the Intel-signed PCK certificate:
    /// whether the platform may add or replace CPU packages after its keys
    /// were registered. Required: there is no default.
    pub dynamic_platform: PckFlag,
    /// Expected `CachedKeys` flag: whether Intel's registration service
    /// keeps the platform's root keys, so a package replacement can recover
    /// them without the original packages present. Required.
    pub cached_keys: PckFlag,
    /// Expected `SMTEnabled` flag: whether simultaneous multithreading is
    /// on, sharing a core's microarchitectural state between a TD and
    /// whatever runs on the sibling thread. Required.
    pub smt_enabled: PckFlag,
}

/// The expected value of a PCK certificate platform flag. Each flag must be
/// stated by the pin author; a quote whose flag differs is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PckFlag {
    /// The certificate asserts the flag.
    True,
    /// The certificate denies the flag.
    False,
    /// The certificate omits the flag, as Processor-CA certificates (single
    /// package platforms) do.
    Undefined,
}

impl AllowedMeasurement {
    /// An AMD SEV-SNP pin.
    pub fn sev_snp(pin: SevSnpPin) -> Self {
        Self {
            platform: PlatformMeasurement::SevSnp(pin),
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
            PlatformMeasurement::SevSnp(_) => Platform::SevSnp,
            PlatformMeasurement::Tdx(_) => Platform::Tdx,
        }
    }
}

/// The launch identity of a SEV-SNP report that matched a pin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SevSnpLaunchMeasurement {
    /// Hex-encoded launch digest.
    pub measurement: String,
    /// Hex-encoded HOST_DATA.
    pub host_data: String,
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
    /// AMD SEV-SNP launch identity, hex-encoded.
    SevSnp(SevSnpLaunchMeasurement),
    /// Intel TDX launch identity, hex-encoded.
    Tdx(TdxLaunchMeasurement),
}

impl fmt::Display for MatchedMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SevSnp(s) => write!(
                f,
                "sev-snp:measurement={},host_data={}",
                s.measurement, s.host_data
            ),
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
    SevSnp(CompiledSevSnpPin),
    Tdx(Box<CompiledTdxPin>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompiledSevSnpPin {
    pub measurement: [u8; 48],
    pub host_data: [u8; 32],
    pub guest_policy: u64,
}

impl CompiledPin {
    pub fn platform(&self) -> Platform {
        match self.platform {
            CompiledPlatform::SevSnp(_) => Platform::SevSnp,
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
    pub ppids: Vec<[u8; 16]>,
    pub dynamic_platform: PckFlag,
    pub cached_keys: PckFlag,
    pub smt_enabled: PckFlag,
}

/// Guest policy bits 0-15: the minimum firmware ABI (major in 15:8, minor in
/// 7:0) the launch requires.
pub(crate) const POLICY_ABI_MASK: u64 = 0xffff;
/// Guest policy bit 17: reserved, must be one.
pub(crate) const POLICY_RESERVED_MBO: u64 = 1 << 17;
/// Guest policy bit 18: a migration agent may be associated with the guest.
pub(crate) const POLICY_MIGRATE_MA: u64 = 1 << 18;
/// Guest policy bit 19: the host may debug (decrypt) the guest.
pub(crate) const POLICY_DEBUG: u64 = 1 << 19;

/// TD_ATTRIBUTES bit 0: the TD is debuggable by the host.
const TD_ATTRIBUTES_DEBUG: u64 = 1 << 0;
/// TD_ATTRIBUTES bit 28: EPT violations are not converted to #VE.
const TD_ATTRIBUTES_SEPT_VE_DISABLE: u64 = 1 << 28;

pub(crate) fn compile_pins(pins: &[AllowedMeasurement]) -> Result<Vec<CompiledPin>, Error> {
    pins.iter()
        .enumerate()
        .map(|(index, pin)| {
            let platform = match &pin.platform {
                PlatformMeasurement::SevSnp(snp) => {
                    CompiledPlatform::SevSnp(compile_sev_snp(snp, index)?)
                }
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

fn compile_sev_snp(pin: &SevSnpPin, index: usize) -> Result<CompiledSevSnpPin, Error> {
    let guest_policy = parse_guest_policy(&pin.guest_policy).ok_or_else(|| {
        Error::InvalidPin(format!(
            "allowed measurement {index}: sev_snp.guest_policy must be 0x followed by lowercase \
             hex with no leading zeros (got {:?})",
            pin.guest_policy
        ))
    })?;
    if guest_policy & (POLICY_DEBUG | POLICY_MIGRATE_MA) != 0
        || guest_policy & POLICY_RESERVED_MBO == 0
    {
        return Err(Error::InvalidPin(format!(
            "allowed measurement {index}: sev_snp.guest_policy must have DEBUG and MIGRATE_MA \
             clear and the reserved bit 17 set"
        )));
    }
    Ok(CompiledSevSnpPin {
        measurement: pin_hex(&pin.measurement, index, "sev_snp.measurement")?,
        host_data: pin_hex(&pin.host_data, index, "sev_snp.host_data")?,
        guest_policy,
    })
}

/// A guest policy in its one spelling: `0x`, then lowercase hex with no
/// leading zeros, fitting 64 bits.
fn parse_guest_policy(text: &str) -> Option<u64> {
    let digits = text.strip_prefix("0x")?;
    let value = u64::from_str_radix(digits, 16).ok()?;
    (format!("{value:#x}") == text).then_some(value)
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
    if policy.ppids.is_empty() {
        return Err(Error::InvalidPin(format!(
            "allowed measurement {index}: tdx.policy.ppids must list at least one machine"
        )));
    }
    let ppids = policy
        .ppids
        .iter()
        .map(|ppid| pin_hex(ppid, index, "tdx.policy.ppids"))
        .collect::<Result<Vec<_>, _>>()?;
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
        ppids,
        dynamic_platform: policy.dynamic_platform,
        cached_keys: policy.cached_keys,
        smt_enabled: policy.smt_enabled,
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
                ppids: vec!["33".repeat(16)],
                dynamic_platform: PckFlag::True,
                cached_keys: PckFlag::True,
                smt_enabled: PckFlag::True,
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

    pub(crate) fn sev_snp_pin() -> SevSnpPin {
        SevSnpPin {
            measurement: "ab".repeat(48),
            host_data: SevSnpPin::host_data_for_config(b"config"),
            guest_policy: "0x30133".to_string(),
        }
    }

    #[test]
    fn host_data_is_the_config_sha256() {
        assert_eq!(
            SevSnpPin::host_data_for_config(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn compile_accepts_a_well_formed_sev_snp_pin() {
        let compiled = compile_pins(&[AllowedMeasurement::sev_snp(sev_snp_pin())]).unwrap();
        let CompiledPlatform::SevSnp(snp) = &compiled[0].platform else {
            panic!("expected a SEV-SNP pin");
        };
        assert_eq!(snp.measurement, [0xab; 48]);
        assert_eq!(snp.guest_policy, 0x30133);
        assert_eq!(compiled[0].platform(), Platform::SevSnp);
    }

    #[test]
    fn compile_rejects_malformed_and_unsafe_sev_snp_pins() {
        type Mutation = Box<dyn Fn(&mut SevSnpPin)>;
        let policy =
            |text: &'static str| -> Mutation { Box::new(move |p| p.guest_policy = text.into()) };
        let cases: Vec<(&str, Mutation)> = vec![
            (
                "sev_snp.measurement",
                Box::new(|p| p.measurement = "ab".repeat(47)),
            ),
            (
                "sev_snp.host_data",
                Box::new(|p| p.host_data = "ab".repeat(48)),
            ),
            (
                "sev_snp.host_data",
                Box::new(|p| p.host_data = "zz".repeat(32)),
            ),
            // One spelling only.
            ("guest_policy must be 0x", policy("0x030133")),
            ("guest_policy must be 0x", policy("0X30133")),
            ("guest_policy must be 0x", policy("0x30133 ")),
            ("guest_policy must be 0x", policy("196915")),
            ("guest_policy must be 0x", policy("0x3013A")),
            ("guest_policy must be 0x", policy("0x10000000000000000")),
            // A policy no sound launch uses: DEBUG, MIGRATE_MA, bit 17 clear.
            ("DEBUG and MIGRATE_MA", policy("0xb0133")),
            ("DEBUG and MIGRATE_MA", policy("0x70133")),
            ("DEBUG and MIGRATE_MA", policy("0x10133")),
        ];
        for (needle, mutate) in cases {
            let mut pin = sev_snp_pin();
            mutate(&mut pin);
            let err = compile_pins(&[AllowedMeasurement::sev_snp(pin.clone())])
                .expect_err(needle)
                .to_string();
            assert!(err.contains(needle), "{needle} ({pin:?}): {err}");
        }
    }

    #[test]
    fn sev_snp_pins_round_trip_through_strict_json() {
        let pin = AllowedMeasurement::sev_snp(sev_snp_pin());
        let json = serde_json::to_value(&pin).unwrap();
        assert_eq!(json["platform"]["sev_snp"]["guest_policy"], "0x30133");
        assert_eq!(
            serde_json::from_value::<AllowedMeasurement>(json.clone()).unwrap(),
            pin
        );
        // Every member is required, and nothing else is accepted.
        for member in ["measurement", "host_data", "guest_policy"] {
            let mut missing = json.clone();
            missing["platform"]["sev_snp"]
                .as_object_mut()
                .unwrap()
                .remove(member);
            let err = serde_json::from_value::<AllowedMeasurement>(missing)
                .unwrap_err()
                .to_string();
            assert!(err.contains(member), "{member}: {err}");
        }
        let mut unknown = json.clone();
        unknown["platform"]["sev_snp"]["rtmr1"] = "00".repeat(48).into();
        assert!(serde_json::from_value::<AllowedMeasurement>(unknown).is_err());
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
            ("ppids must list", Box::new(|p| p.policy.ppids.clear())),
            (
                "tdx.policy.ppids",
                Box::new(|p| p.policy.ppids = vec!["00".repeat(15)]),
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

        assert_eq!(json["platform"]["tdx"]["policy"]["smt_enabled"], "true");
        let mut unknown = json.clone();
        unknown["platform"]["tdx"]["rtmr1"] = serde_json::Value::String("00".repeat(48));
        assert!(serde_json::from_value::<AllowedMeasurement>(unknown).is_err());

        // Each platform flag, and the machine allowlist, is required:
        // omitting one is not a default.
        for flag in ["dynamic_platform", "cached_keys", "smt_enabled", "ppids"] {
            let mut missing = json.clone();
            missing["platform"]["tdx"]["policy"]
                .as_object_mut()
                .unwrap()
                .remove(flag);
            let err = serde_json::from_value::<AllowedMeasurement>(missing)
                .unwrap_err()
                .to_string();
            assert!(err.contains(flag), "{flag}: {err}");
        }
    }
}
