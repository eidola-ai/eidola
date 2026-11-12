//! SEV-SNP attestation verification using the `sev` crate.
//!
//! Delegates certificate chain verification and report signature checking to
//! the [`sev`](https://crates.io/crates/sev) crate (virtee/sev). This crate
//! adds TCB policy enforcement (configurable per-component floors plus a
//! rollback check against `committed_tcb`) and TLS fingerprint
//! cross-checking on top.
//!
//! ## Processor generations
//!
//! Two AMD EPYC generations are accepted, Genoa and Turin, each with its own
//! AMD root (ARK) and intermediate (ASK), both built into the `sev` crate
//! (`builtin::genoa`, `builtin::turin`). The chain is chosen by the
//! generation the report itself names ([`SevSnpGeneration::of_report`]):
//! the CPUID family and model a version-3 or later report carries, inside
//! the signed body. It is never chosen by trying each chain. A Turin report
//! verified against the Genoa chain, or the reverse, fails at the chain.
//! The generation also fixes the report's TCB_VERSION layout (Turin's has a
//! fifth component, FMC, and a different byte order), the VCEK extensions
//! the report is bound to, the length of the chip identity, and the TCB
//! floor ([`SevSnpTcbFloors`]).
//!
//! A version-2 report carries no CPUID fields. [`check_report_hygiene`]
//! refuses it before any chain is chosen, on every generation, so no
//! generation is ever inferred from anything but the signed CPUID.
//!
//! ## TCB policy: AMD vs Intel
//!
//! AMD does not publish a pull-based TCB recommendation feed equivalent to
//! Intel's `tcb_info` JSON, so the relying party's "minimum TCB SVNs" are
//! operator-set rather than fetched from a remote endpoint. The defaults
//! ([`SevSnpTcbPolicy::genoa`], [`SevSnpTcbPolicy::turin`]) are the
//! `minimum_tcb` of the platform provider's SEV-SNP policies for each
//! generation, as its signed platform endorsements state them; operators
//! can tighten them via [`crate::AttestingClientConfig::snp_min_tcb`].
//!
//! ## Rollback protection
//!
//! Each SEV-SNP attestation report carries multiple TCB version fields.
//! The two we care about for policy purposes are:
//!
//! - `reported_tcb`: the TCB version associated with the VCEK that signed
//!   the report. The hypervisor can change this via the firmware's
//!   `SET_TCB_VERSION` command, but only downward (it can't lie upward,
//!   because there is no VCEK that would sign for a higher TCB — a fact
//!   [`verify_report`] enforces by comparing `reported_tcb` and `CHIP_ID`
//!   with the SPL and hwID extensions of the VCEK that signed the report;
//!   the VCEK arrives as untrusted collateral, so the signature alone binds
//!   the report to *some* genuine VCEK, not to this TCB).
//! - `committed_tcb`: a one-way commit by the firmware. Once the firmware
//!   commits to a TCB level, it will not honor any `SET_TCB_VERSION`
//!   request that would drop reported_tcb below it.
//!
//! A malicious hypervisor that wants to make an enclave appear to be
//! running on an older (and known-vulnerable) TCB could call
//! `SET_TCB_VERSION` to drop `reported_tcb` to a value the firmware has
//! *not* committed never to honor. We catch this by requiring
//! `reported_tcb >= committed_tcb` componentwise; the only legitimate way
//! the inequality could fail is a firmware bug or a hypervisor that's
//! actively lying about the TCB level. We classify this as a separate,
//! more severe failure mode than "below operator floor."

use std::io::Cursor;
use std::sync::Arc;

use der::{Decode, Encode};
use sev::certs::snp::{
    Certificate, Chain, Verifiable,
    builtin::{genoa, turin},
    ca,
};
use sev::firmware::guest::AttestationReport;
use sev::firmware::host::TcbVersion;
use sev::parser::Decoder;
use sha2::{Digest, Sha256};

use crate::Error;

/// Exact wire size of a SEV-SNP attestation report (AMD SEV-SNP ABI,
/// `ATTESTATION_REPORT` structure). The `sev` crate reads exactly this many
/// bytes and would silently ignore trailing data, so the length is pinned
/// here to keep the accepted encoding unique.
const REPORT_LEN: usize = 1184;

/// Parse a raw attestation report without verifying its signature.
pub fn parse_report(report_bytes: &[u8]) -> Result<AttestationReport, Error> {
    if report_bytes.len() != REPORT_LEN {
        return Err(Error::Report(format!(
            "attestation report must be exactly {REPORT_LEN} bytes, got {}",
            report_bytes.len()
        )));
    }
    AttestationReport::decode(&mut Cursor::new(report_bytes), ())
        .map_err(|e| Error::Report(format!("failed to parse attestation report: {e}")))
}

/// Structural checks on report fields that are *not* covered by the launch
/// measurement and must therefore be policed by the relying party.
///
/// The guest policy is the sharp one: it is chosen by the hypervisor at
/// launch and enforced by the PSP, but it is **not** folded into the launch
/// digest — a malicious host can relaunch the exact pinned image with
/// `POLICY.DEBUG=1` and obtain an identical measurement while gaining
/// `SNP_DBG_DECRYPT` access to guest memory. `MIGRATE_MA` similarly hands
/// guest state to a migration agent outside the measured image. Both must
/// be off.
///
/// The signer/ID-block fields are hygiene: the report must be VCEK-signed
/// (`SIGNING_KEY=0`), with the chip identity unmasked and no ID-block or
/// author key in play — matching what Tinfoil's own v3 verifier enforces
/// and what its production fleet presents.
///
/// Report version 3 (firmware 1.55+) is required so the CPUID
/// family/model/stepping fields are present and the product generation is
/// bound by the report itself rather than inferred. A version-2 report is
/// refused on every generation: the `sev` crate would otherwise guess its
/// generation (and so its TCB layout) from the shape of CHIP_ID.
pub fn check_report_hygiene(report: &AttestationReport) -> Result<(), Error> {
    if report.version < 3 {
        return Err(Error::Report(format!(
            "attestation report version {} is below the required minimum of 3",
            report.version
        )));
    }
    if report.policy.debug_allowed() {
        return Err(Error::Report(
            "guest policy allows DEBUG: the hypervisor could decrypt guest memory \
             via SNP_DBG_DECRYPT despite a matching launch measurement"
                .to_string(),
        ));
    }
    if report.policy.migrate_ma_allowed() {
        return Err(Error::Report(
            "guest policy allows migration-agent association".to_string(),
        ));
    }
    if report.key_info.mask_chip_key() {
        return Err(Error::Report(
            "report has MASK_CHIP_KEY set; chip identity is masked".to_string(),
        ));
    }
    if report.key_info.signing_key() != 0 {
        return Err(Error::Report(format!(
            "report SIGNING_KEY is {}; only VCEK (0) is accepted",
            report.key_info.signing_key()
        )));
    }
    if report.key_info.author_key_en()
        || report.id_key_digest != [0u8; 48]
        || report.author_key_digest != [0u8; 48]
    {
        return Err(Error::Report(
            "report indicates an ID-block launch (AUTHOR_KEY_EN or a non-zero \
             ID_KEY_DIGEST/AUTHOR_KEY_DIGEST); only plain launches are accepted"
                .to_string(),
        ));
    }
    // GUEST_SVN, FAMILY_ID and IMAGE_ID come only from an ID block, so a
    // plain launch reports them zero; anything else is not the launch the
    // checks above describe.
    if report.guest_svn != 0 || report.family_id != [0u8; 16] || report.image_id != [0u8; 16] {
        return Err(Error::Report(
            "report carries a GUEST_SVN, FAMILY_ID or IMAGE_ID, which only an ID block sets"
                .to_string(),
        ));
    }
    // The report was requested from VMPL 0, the guest's most privileged
    // level: the measured image runs there and runs no lower-privileged
    // layer that could ask for reports of its own.
    if report.vmpl != 0 {
        return Err(Error::Report(format!(
            "report was requested from VMPL {}; only VMPL 0 is accepted",
            report.vmpl
        )));
    }
    Ok(())
}

/// The guest policy exactly as the report carries it (`POLICY`, bytes
/// 0x08..0x10, little-endian), without the `sev` crate's normalization of
/// the reserved bit.
pub(crate) fn raw_policy(report_bytes: &[u8]) -> Result<u64, Error> {
    report_bytes
        .get(0x08..0x10)
        .and_then(|b| b.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| Error::Report("attestation report is too short for POLICY".to_string()))
}

/// Appraise a report's guest policy against a pin's: every bit equal except
/// the firmware ABI minimum (bits 0-15, major in 15:8 and minor in 7:0),
/// which may be higher than the pin's, since a launch that asks for newer
/// firmware than the image needs is not weaker for it. The pin's own policy
/// has DEBUG and MIGRATE_MA clear (`measurement::compile_pins`), so equality
/// refuses both, and refuses a launch that drops a restriction the image
/// asks for or adds a permission it does not (CXL, for instance).
pub(crate) fn check_guest_policy(reported: u64, pinned: u64) -> Result<(), Error> {
    use crate::measurement::POLICY_ABI_MASK;
    if reported & !POLICY_ABI_MASK != pinned & !POLICY_ABI_MASK {
        return Err(Error::Report(format!(
            "guest policy {reported:#x} differs from the pinned {pinned:#x} outside the \
             firmware ABI minimum"
        )));
    }
    let abi = |policy: u64| ((policy >> 8) & 0xff, policy & 0xff);
    if abi(reported) < abi(pinned) {
        return Err(Error::Report(format!(
            "guest policy {reported:#x} allows older firmware than the pinned {pinned:#x}"
        )));
    }
    Ok(())
}

/// The AMD EPYC generation an attestation report was produced on, as the
/// report's signed CPUID fields name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SevSnpGeneration {
    /// Fourth-generation EPYC (CPUID family 19h).
    Genoa,
    /// Fifth-generation EPYC (CPUID family 1Ah).
    Turin,
}

impl SevSnpGeneration {
    /// The generation a parsed report names.
    ///
    /// Read from the report's CPUID family and model (present from report
    /// version 3), mapped by the `sev` crate's `Generation::identify_cpu`,
    /// the same mapping its parser used to choose the report's TCB_VERSION
    /// layout, so the chain, the layout and the floors all follow one
    /// reading of one signed field. A report without CPUID fields, or of any
    /// other generation (Milan, Venice, an unknown model), is refused: no
    /// chain for it is pinned.
    pub fn of_report(report: &AttestationReport) -> Result<Self, Error> {
        let (Some(family), Some(model)) = (report.cpuid_fam_id, report.cpuid_mod_id) else {
            return Err(Error::Report(format!(
                "attestation report version {} carries no CPUID family and model, so it \
                 names no processor generation",
                report.version
            )));
        };
        let generation = match sev::Generation::identify_cpu(family, model) {
            Ok(sev::Generation::Genoa) => Self::Genoa,
            Ok(sev::Generation::Turin) => Self::Turin,
            _ => {
                return Err(Error::Report(format!(
                    "report CPUID family {family:#04x} model {model:#04x} is neither Genoa \
                     nor Turin; no AMD chain is pinned for it"
                )));
            }
        };
        // The parser read every TCB_VERSION in the layout of the generation it
        // took from these same bytes; only Turin's layout has an FMC byte.
        let turin_layout = generation == Self::Turin;
        if [report.reported_tcb, report.committed_tcb]
            .iter()
            .any(|tcb| tcb.fmc.is_some() != turin_layout)
        {
            return Err(Error::Report(format!(
                "report TCB layout does not match its {generation} CPUID"
            )));
        }
        Ok(generation)
    }

    /// The generation's built-in AMD ARK and ASK.
    fn builtin_chain(self) -> Result<(Certificate, Certificate), Error> {
        let (ark, ask) = match self {
            Self::Genoa => (genoa::ark(), genoa::ask()),
            Self::Turin => (turin::ark(), turin::ask()),
        };
        let ark = ark.map_err(|e| Error::CertChain(format!("failed to load {self} ARK: {e}")))?;
        let ask = ask.map_err(|e| Error::CertChain(format!("failed to load {self} ASK: {e}")))?;
        Ok((ark, ask))
    }

    /// Significant length of the generation's chip identity: the VCEK's
    /// hwID extension, and the leading bytes of a report's 64-byte CHIP_ID
    /// (the rest are zero on Turin).
    fn hwid_len(self) -> usize {
        match self {
            Self::Genoa => 64,
            Self::Turin => 8,
        }
    }
}

impl std::fmt::Display for SevSnpGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Genoa => "Genoa",
            Self::Turin => "Turin",
        })
    }
}

/// Verify a VCEK certificate chain and an already-parsed report's signature.
///
/// 1. Builds the chain: custom ARK or the generation's built-in ARK →
///    custom ASK or the generation's built-in ASK → VCEK
/// 2. Verifies the chain (ARK self-signed, ARK signs ASK, ASK signs VCEK)
/// 3. Verifies the report's ECDSA-P384 signature against the VCEK
/// 4. Binds the report to the VCEK (`check_vcek_binding`)
///
/// `generation` is the report's own ([`SevSnpGeneration::of_report`]). A
/// custom certificate replaces the built-in one whatever the generation, so
/// an override is one chain, never a chain added beside AMD's.
pub fn verify_report(
    vcek_der: &[u8],
    report: &AttestationReport,
    generation: SevSnpGeneration,
    ark_der: Option<&[u8]>,
    ask_der: Option<&[u8]>,
) -> Result<(), Error> {
    let (builtin_ark, builtin_ask) = generation.builtin_chain()?;
    let ark = match ark_der {
        Some(der) => Certificate::from_der(der)
            .map_err(|e| Error::CertChain(format!("failed to parse custom ARK: {e}")))?,
        None => builtin_ark,
    };

    let ask = match ask_der {
        Some(der) => Certificate::from_der(der)
            .map_err(|e| Error::CertChain(format!("failed to parse custom ASK: {e}")))?,
        None => builtin_ask,
    };

    let chain = Chain {
        ca: ca::Chain { ark, ask },
        vek: Certificate::from_der(vcek_der)
            .map_err(|e| Error::CertChain(format!("failed to parse VCEK cert: {e}")))?,
    };

    tracing::debug!(%generation, "Verifying VCEK certificate chain...");
    let verified_vek = (&chain)
        .verify()
        .map_err(|e| Error::CertChain(format!("certificate chain verification failed: {e}")))?;

    tracing::debug!("Verifying attestation report signature...");
    (verified_vek, report)
        .verify()
        .map_err(|e| Error::Signature(format!("report signature verification failed: {e}")))?;

    check_vcek_binding(vcek_der, report, generation)?;

    Ok(())
}

/// AMD KDS VCEK extension OIDs (VCEK/VLEK certificate spec, AMD 57230;
/// the same set `go-sev-guest`'s `kds` package reads, plus the FMC SPL that
/// Turin VCEKs carry and `snpguest` reads).
const OID_BL_SPL: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.3.1");
const OID_TEE_SPL: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.3.2");
const OID_SNP_SPL: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.3.3");
const OID_UCODE_SPL: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.3.8");
const OID_FMC_SPL: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.3.9");
const OID_HWID: der::asn1::ObjectIdentifier =
    der::asn1::ObjectIdentifier::new_unwrap("1.3.6.1.4.1.3704.1.4");

/// Bind the report to the VCEK that signed it.
///
/// A VCEK is derived per chip *and per TCB*: its certificate's SPL
/// extensions name the TCB it was endorsed for, and its HWID names the
/// chip. The signature alone proves only that *some* genuine VCEK signed
/// the report, and the VCEK arrives as untrusted collateral the document's
/// author chooses — so without this check, a VCEK whose key leaked from a
/// vulnerable TCB could sign a report claiming any `reported_tcb`, and
/// every floor in [`SevSnpTcbPolicy`] would be enforced against a number
/// the attacker wrote. This is what makes the module doc's "there is no
/// VCEK that would sign for a higher TCB" true. Mirrors `go-sev-guest`'s
/// `validateTcb` (`reported == cert`) and its CHIP_ID/HWID check, including
/// its tolerance for the KDS encoding the HWID as untagged raw bytes.
///
/// Per generation: a Turin VCEK must carry an fmcSPL extension, compared
/// with the report's FMC SVN, and a Genoa VCEK must not carry one. The hwID
/// is 64 bytes on Genoa and 8 on Turin, and is compared with the whole
/// 64-byte CHIP_ID, read as the hwID followed by zeros (how Turin firmware
/// writes it, and how the platform provider's endorsements list Turin
/// chips), so no byte of CHIP_ID goes uncompared.
fn check_vcek_binding(
    vcek_der: &[u8],
    report: &AttestationReport,
    generation: SevSnpGeneration,
) -> Result<(), Error> {
    let cert = x509_cert::Certificate::from_der(vcek_der)
        .map_err(|e| Error::CertParse(format!("failed to parse VCEK DER: {e}")))?;
    let extensions = cert.tbs_certificate.extensions.as_deref().unwrap_or(&[]);
    let find_optional =
        |oid: der::asn1::ObjectIdentifier, name: &str| -> Result<Option<&[u8]>, Error> {
            let mut found = extensions.iter().filter(|ext| ext.extn_id == oid);
            let ext = found.next();
            if found.next().is_some() {
                return Err(Error::CertChain(format!(
                    "VCEK certificate carries duplicate {name} extensions"
                )));
            }
            Ok(ext.map(|ext| ext.extn_value.as_bytes()))
        };
    let find = |oid: der::asn1::ObjectIdentifier, name: &str| -> Result<&[u8], Error> {
        find_optional(oid, name)?.ok_or_else(|| {
            Error::CertChain(format!("VCEK certificate carries no {name} extension"))
        })
    };
    let spl = |oid, name| -> Result<u8, Error> {
        u8::from_der(find(oid, name)?).map_err(|e| {
            Error::CertChain(format!(
                "VCEK {name} extension is not a DER INTEGER 0-255: {e}"
            ))
        })
    };

    let fmc = match generation {
        SevSnpGeneration::Turin => Some(spl(OID_FMC_SPL, "fmcSPL")?),
        SevSnpGeneration::Genoa => {
            if find_optional(OID_FMC_SPL, "fmcSPL")?.is_some() {
                return Err(Error::CertChain(
                    "Genoa VCEK certificate carries an fmcSPL extension".to_string(),
                ));
            }
            None
        }
    };
    let cert_tcb = SevSnpTcbSvns {
        bootloader: spl(OID_BL_SPL, "blSPL")?,
        tee: spl(OID_TEE_SPL, "teeSPL")?,
        snp: spl(OID_SNP_SPL, "snpSPL")?,
        microcode: spl(OID_UCODE_SPL, "ucodeSPL")?,
        fmc,
    };
    let reported = SevSnpTcbSvns::from(report.reported_tcb);
    if cert_tcb != reported {
        return Err(Error::CertChain(format!(
            "report reported_tcb ({reported}) is not the TCB its VCEK was endorsed for \
             ({cert_tcb})"
        )));
    }

    let hwid_len = generation.hwid_len();
    let hwid_ext = find(OID_HWID, "hwID")?;
    let hwid = if hwid_ext.len() == hwid_len {
        hwid_ext
    } else {
        der::asn1::OctetStringRef::from_der(hwid_ext)
            .map(|octets| octets.as_bytes())
            .ok()
            .filter(|bytes| bytes.len() == hwid_len)
            .ok_or_else(|| {
                Error::CertChain(format!(
                    "{generation} VCEK hwID extension is not {hwid_len} bytes"
                ))
            })?
    };
    let mut expected_chip_id = [0u8; 64];
    expected_chip_id[..hwid_len].copy_from_slice(hwid);
    // An all-zero CHIP_ID is a masked chip identity (MASK_CHIP_ID); there is
    // nothing to compare, as in go-sev-guest. The TCB binding above still
    // holds.
    if report.chip_id != [0u8; 64] && report.chip_id != expected_chip_id {
        return Err(Error::CertChain(
            "report CHIP_ID is not the hwID of the VCEK that signed it".to_string(),
        ));
    }

    Ok(())
}

/// Resolve ARK and ASK as raw DER bytes: each custom certificate when
/// supplied, otherwise the built-in AMD certificate of the report's
/// generation.
///
/// Used by the per-handshake connector to obtain the DER bytes it needs
/// for downstream operations the `sev` crate's `verify_report` doesn't
/// itself return — specifically, parsing into `x509-cert::Certificate`
/// for CRL signature verification and serial number extraction.
pub fn resolve_chain_certs_der(
    generation: SevSnpGeneration,
    custom_ark: Option<&[u8]>,
    custom_ask: Option<&[u8]>,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let (builtin_ark, builtin_ask) = generation.builtin_chain()?;
    let encode = |cert: Certificate, name: &str| {
        cert.to_der()
            .map_err(|e| Error::CertChain(format!("failed to DER-encode {generation} {name}: {e}")))
    };
    let ark = match custom_ark {
        Some(der) => der.to_vec(),
        None => encode(builtin_ark, "ARK")?,
    };
    let ask = match custom_ask {
        Some(der) => der.to_vec(),
        None => encode(builtin_ask, "ASK")?,
    };
    Ok((ark, ask))
}

/// Extract the raw serial number bytes from a DER-encoded X.509
/// certificate. The result is suitable for direct comparison against
/// the serial numbers in an `x509_cert::crl::CertificateList`.
pub fn cert_serial_from_der(cert_der: &[u8]) -> Result<Vec<u8>, Error> {
    let cert = x509_cert::Certificate::from_der(cert_der)
        .map_err(|e| Error::CertParse(format!("failed to parse cert DER: {e}")))?;
    Ok(cert.tbs_certificate.serial_number.as_bytes().to_vec())
}

/// Per-component TCB SVNs extracted from a SEV-SNP attestation report.
///
/// We re-define this rather than re-exporting [`sev::firmware::host::TcbVersion`]
/// so the public surface of `tinfoil-verifier` does not leak its dependency
/// on the `sev` crate. `fmc` is the fifth component Turin's TCB_VERSION
/// carries; it is bound to the VCEK, held to the rollback rule and to the
/// floor like the other four.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SevSnpTcbSvns {
    pub bootloader: u8,
    pub tee: u8,
    pub snp: u8,
    pub microcode: u8,
    /// Present on Turin; `None` on Genoa, whose TCB_VERSION has no FMC.
    pub fmc: Option<u8>,
}

impl From<TcbVersion> for SevSnpTcbSvns {
    fn from(t: TcbVersion) -> Self {
        Self {
            bootloader: t.bootloader,
            tee: t.tee,
            snp: t.snp,
            microcode: t.microcode,
            fmc: t.fmc,
        }
    }
}

impl std::fmt::Display for SevSnpTcbSvns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bl={:#04x} tee={:#04x} snp={:#04x} ucode={:#04x}",
            self.bootloader, self.tee, self.snp, self.microcode,
        )?;
        if let Some(fmc) = self.fmc {
            write!(f, " fmc={fmc:#04x}")?;
        }
        Ok(())
    }
}

/// Minimum TCB SVNs for each accepted processor generation.
///
/// TCB SVNs are numbered per generation (a Turin bootloader SPL of 1 and a
/// Genoa one of 7 say nothing about each other), so each generation has its
/// own floor, and a report is held to the floor of the generation its CPUID
/// names. Defaults: [`SevSnpTcbPolicy::genoa`] and
/// [`SevSnpTcbPolicy::turin`]; operators can tighten either through
/// [`crate::AttestingClientConfig::snp_min_tcb`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SevSnpTcbFloors {
    pub genoa: SevSnpTcbPolicy,
    pub turin: SevSnpTcbPolicy,
}

impl SevSnpTcbFloors {
    /// The floor a report of `generation` is held to.
    pub fn for_generation(&self, generation: SevSnpGeneration) -> &SevSnpTcbPolicy {
        match generation {
            SevSnpGeneration::Genoa => &self.genoa,
            SevSnpGeneration::Turin => &self.turin,
        }
    }

    /// Evaluate a report against its generation's floor
    /// ([`SevSnpTcbPolicy::evaluate`]).
    pub fn evaluate(
        &self,
        report: &AttestationReport,
        generation: SevSnpGeneration,
    ) -> (SevSnpTcbObservation, Result<(), Error>) {
        self.for_generation(generation).evaluate(report, generation)
    }
}

impl Default for SevSnpTcbFloors {
    fn default() -> Self {
        Self {
            genoa: SevSnpTcbPolicy::genoa(),
            turin: SevSnpTcbPolicy::turin(),
        }
    }
}

/// Minimum TCB SVNs the verifier will accept from one processor generation.
///
/// `min_fmc` applies to the FMC component, which only Turin's TCB carries;
/// a report without one is read as FMC 0 against it, so a non-zero
/// `min_fmc` refuses every Genoa report rather than being ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SevSnpTcbPolicy {
    pub min_fmc: u8,
    pub min_bootloader: u8,
    pub min_tee: u8,
    pub min_snp: u8,
    pub min_microcode: u8,
}

impl SevSnpTcbPolicy {
    /// Genoa's floor: `bootloader >= 7`, `tee >= 0`, `snp >= 14`,
    /// `microcode >= 72`. These are the `minimum_tcb` of the `amd-genoa-prod`
    /// SEV-SNP policy in the platform provider's signed endorsements
    /// (vendored under `releases/trust/platform-endorsements/`, release
    /// `platform-v0.1.11`), and the floor `go-sev-guest` enforces for Genoa.
    pub fn genoa() -> Self {
        Self {
            min_fmc: 0,
            min_bootloader: 7,
            min_tee: 0,
            min_snp: 14,
            min_microcode: 72,
        }
    }

    /// Turin's floor: `fmc >= 1`, `bootloader >= 1`, `tee >= 1`,
    /// `snp >= 4`, `microcode >= 82`. These are the `minimum_tcb` of the
    /// `amd-turin-prod` SEV-SNP policy in the platform provider's signed
    /// endorsements (vendored under `releases/trust/platform-endorsements/`,
    /// release `platform-v0.1.11`).
    pub fn turin() -> Self {
        Self {
            min_fmc: 1,
            min_bootloader: 1,
            min_tee: 1,
            min_snp: 4,
            min_microcode: 82,
        }
    }

    /// Evaluate a parsed attestation report against this policy.
    ///
    /// Returns the [`SevSnpTcbObservation`] (always — even on failure, so
    /// observers can record rejected attestations) and a `Result` that is
    /// `Ok(())` when the report passes, or `Err(Error::TcbPolicy)` when
    /// either the rollback check fails or any SVN is below the configured
    /// floor. Rollback is checked before the floor so the error message
    /// reflects the more severe condition first.
    pub fn evaluate(
        &self,
        report: &AttestationReport,
        generation: SevSnpGeneration,
    ) -> (SevSnpTcbObservation, Result<(), Error>) {
        let reported = SevSnpTcbSvns::from(report.reported_tcb);
        let committed = SevSnpTcbSvns::from(report.committed_tcb);

        let rollback = self.detect_rollback(&reported, &committed);
        let below_floor = self.detect_below_floor(&reported);

        let bucket = if rollback.is_some() {
            BUCKET_ROLLBACK
        } else if below_floor.is_some() {
            BUCKET_BELOW_FLOOR
        } else {
            BUCKET_MEETS_FLOOR
        };

        let observation = SevSnpTcbObservation {
            generation,
            reported_tcb: reported,
            committed_tcb: committed,
            chip_id: report.chip_id,
            bucket,
        };

        let result = match (rollback, below_floor) {
            (Some(msg), _) => Err(Error::TcbPolicy(msg)),
            (None, Some(msg)) => Err(Error::TcbPolicy(msg)),
            (None, None) => Ok(()),
        };

        (observation, result)
    }

    fn detect_rollback(
        &self,
        reported: &SevSnpTcbSvns,
        committed: &SevSnpTcbSvns,
    ) -> Option<String> {
        if reported.fmc.unwrap_or(0) < committed.fmc.unwrap_or(0)
            || reported.bootloader < committed.bootloader
            || reported.tee < committed.tee
            || reported.snp < committed.snp
            || reported.microcode < committed.microcode
        {
            Some(format!(
                "SEV-SNP reported_tcb ({reported}) is below committed_tcb ({committed}); \
                 possible firmware rollback or hypervisor SET_TCB_VERSION abuse",
            ))
        } else {
            None
        }
    }

    fn detect_below_floor(&self, reported: &SevSnpTcbSvns) -> Option<String> {
        let mut violations: Vec<String> = Vec::new();
        let fmc = reported.fmc.unwrap_or(0);
        if fmc < self.min_fmc {
            violations.push(format!("fmc {fmc:#04x} < min {:#04x}", self.min_fmc));
        }
        if reported.bootloader < self.min_bootloader {
            violations.push(format!(
                "bootloader {:#04x} < min {:#04x}",
                reported.bootloader, self.min_bootloader
            ));
        }
        if reported.tee < self.min_tee {
            violations.push(format!(
                "tee {:#04x} < min {:#04x}",
                reported.tee, self.min_tee
            ));
        }
        if reported.snp < self.min_snp {
            violations.push(format!(
                "snp {:#04x} < min {:#04x}",
                reported.snp, self.min_snp
            ));
        }
        if reported.microcode < self.min_microcode {
            violations.push(format!(
                "microcode {:#04x} < min {:#04x}",
                reported.microcode, self.min_microcode
            ));
        }
        if violations.is_empty() {
            None
        } else {
            Some(format!(
                "SEV-SNP reported_tcb below operator floor: {}",
                violations.join(", "),
            ))
        }
    }
}

// Bucket labels for `SevSnpTcbObservation::as_metric_label`. Stable
// strings — dashboards and alert rules depend on these.
const BUCKET_MEETS_FLOOR: &str = "meets_floor";
const BUCKET_BELOW_FLOOR: &str = "below_floor";
const BUCKET_ROLLBACK: &str = "rollback_detected";

/// Observation surfaced after a SEV-SNP attestation has been
/// signature-verified, before the policy result is propagated.
///
/// Consumers receive this via the optional observer callback on
/// [`crate::AttestingClientConfig`] and can use it to drive metrics,
/// traces, or alerting. The observer fires for *every* attestation that
/// completes signature verification, including ones the policy
/// subsequently rejects, so operators have full visibility into the
/// population of observed TCB levels — not just the ones that made it
/// through.
#[derive(Debug, Clone)]
pub struct SevSnpTcbObservation {
    /// Processor generation the report's CPUID names; the floor applied was
    /// this generation's.
    pub generation: SevSnpGeneration,
    /// TCB version associated with the VCEK that signed the report.
    pub reported_tcb: SevSnpTcbSvns,
    /// TCB version the firmware has one-way-committed to. The verifier
    /// requires `reported_tcb >= committed_tcb` componentwise.
    pub committed_tcb: SevSnpTcbSvns,
    /// 64-byte chip identifier from the report. High cardinality —
    /// suitable for trace enrichment, *not* as a metric label.
    pub chip_id: [u8; 64],
    bucket: &'static str,
}

impl SevSnpTcbObservation {
    /// Stable lowercase identifier suitable for use as a metric label.
    /// One of `meets_floor`, `below_floor`, or `rollback_detected`.
    pub fn as_metric_label(&self) -> &'static str {
        self.bucket
    }
}

/// Observer callback type. Invoked synchronously inside the connector
/// layer for every SEV-SNP attestation that completes signature
/// verification, regardless of policy outcome. Implementations must be
/// cheap and non-blocking — they run on the TLS handshake hot path.
pub type SevSnpObserver = Arc<dyn Fn(&SevSnpTcbObservation) + Send + Sync>;

/// Compute SHA-256 of the SPKI from a raw DER-encoded certificate.
pub fn sha256_spki_from_der(cert_der: &[u8]) -> Result<[u8; 32], Error> {
    let cert = x509_cert::Certificate::from_der(cert_der)
        .map_err(|e| Error::CertParse(format!("failed to parse cert DER: {e}")))?;
    let spki_der = cert
        .tbs_certificate
        .subject_public_key_info
        .to_der()
        .map_err(|e| Error::CertParse(format!("failed to encode SPKI to DER: {e}")))?;
    Ok(Sha256::digest(&spki_der).into())
}

/// Reports and VCEKs here are constructed. No genuine Turin report or VCEK
/// is available offline (the platform provider's conformance vectors carry
/// no licence and are not committed), so the Turin cases run on reports
/// built byte by byte and on synthetic VCEKs that nothing signed:
/// `verify_report` checks the signature before the binding, and these tests
/// hold the logic after it (generation, TCB layout, FMC and hwID binding,
/// floors). That a genuine Turin VCEK carries exactly the extensions assumed
/// here (fmcSPL at `1.3.6.1.4.1.3704.1.3.9`, an 8-byte hwID) follows AMD's
/// KDS as `snpguest` reads it; it is not observed in this tree. The two
/// built-in chains themselves are AMD's real certificates.
#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The built-in chains are AMD's published ARK and ASK for each
    /// generation, pinned by the SHA-256 of their DER. The Turin pair, as
    /// the `sev` crate embeds it, is byte-identical to the one
    /// `go-sev-guest` embeds from `https://kdsintf.amd.com/vcek/v1/Turin/cert_chain`.
    #[test]
    fn the_builtin_chains_are_amds_genoa_and_turin_certificates() {
        for (generation, ark_sha256, ask_sha256, ark_cn, ask_cn) in [
            (
                SevSnpGeneration::Genoa,
                "4c6598d19c18719c5dfd4a7d335f674e5bfe1d8f800cea2cf270c10d103db2f1",
                "5464738c1546aed5f2cecf1dc98c5c960a92e8913238a61711bc90ec6e828521",
                "CN=ARK-Genoa",
                "CN=SEV-Genoa",
            ),
            (
                SevSnpGeneration::Turin,
                "1f084161a44bb6d93778a904877d4819cafa5d05ef4193b2ded9dd9c73dd3f6a",
                "5b77ef5fe7a7a004fd9032668fba9d0fda22f88c4442069a479636a6ae3b3185",
                "CN=ARK-Turin",
                "CN=SEV-Turin",
            ),
        ] {
            let (ark, ask) = resolve_chain_certs_der(generation, None, None).unwrap();
            assert_eq!(
                hex::encode(Sha256::digest(&ark)),
                ark_sha256,
                "{generation}"
            );
            assert_eq!(
                hex::encode(Sha256::digest(&ask)),
                ask_sha256,
                "{generation}"
            );
            let subject = |der: &[u8]| {
                x509_cert::Certificate::from_der(der)
                    .unwrap()
                    .tbs_certificate
                    .subject
                    .to_string()
            };
            assert!(subject(&ark).contains(ark_cn), "{}", subject(&ark));
            assert!(subject(&ask).contains(ask_cn), "{}", subject(&ask));
            let (ark, ask) = generation.builtin_chain().unwrap();
            (&ca::Chain { ark, ask })
                .verify()
                .unwrap_or_else(|e| panic!("{generation} CA chain: {e}"));
        }
    }

    fn report_with_tcb(reported: SevSnpTcbSvns, committed: SevSnpTcbSvns) -> AttestationReport {
        AttestationReport {
            reported_tcb: TcbVersion {
                bootloader: reported.bootloader,
                tee: reported.tee,
                snp: reported.snp,
                microcode: reported.microcode,
                fmc: reported.fmc,
            },
            committed_tcb: TcbVersion {
                bootloader: committed.bootloader,
                tee: committed.tee,
                snp: committed.snp,
                microcode: committed.microcode,
                fmc: committed.fmc,
            },
            ..Default::default()
        }
    }

    fn svns(bootloader: u8, tee: u8, snp: u8, microcode: u8) -> SevSnpTcbSvns {
        SevSnpTcbSvns {
            bootloader,
            tee,
            snp,
            microcode,
            fmc: None,
        }
    }

    #[test]
    fn policy_accepts_report_at_or_above_floor() {
        let policy = SevSnpTcbPolicy::genoa();
        let reported = svns(0x07, 0x00, 0x0E, 0x48);
        let committed = svns(0x07, 0x00, 0x0E, 0x48);
        let report = report_with_tcb(reported, committed);
        let (obs, result) = policy.evaluate(&report, SevSnpGeneration::Genoa);
        assert!(result.is_ok());
        assert_eq!(obs.as_metric_label(), "meets_floor");
    }

    #[test]
    fn policy_accepts_report_above_floor() {
        let policy = SevSnpTcbPolicy::genoa();
        let reported = svns(0x10, 0x05, 0x20, 0x80);
        let committed = svns(0x10, 0x05, 0x20, 0x80);
        let (obs, result) = policy.evaluate(
            &report_with_tcb(reported, committed),
            SevSnpGeneration::Genoa,
        );
        assert!(result.is_ok());
        assert_eq!(obs.as_metric_label(), "meets_floor");
    }

    #[test]
    fn policy_rejects_below_bootloader_floor() {
        let policy = SevSnpTcbPolicy::genoa();
        let reported = svns(0x06, 0x00, 0x0E, 0x48); // bootloader one short
        let committed = svns(0x06, 0x00, 0x0E, 0x48); // committed matches, no rollback
        let (obs, result) = policy.evaluate(
            &report_with_tcb(reported, committed),
            SevSnpGeneration::Genoa,
        );
        let err = result.unwrap_err();
        assert!(matches!(err, Error::TcbPolicy(_)));
        assert_eq!(obs.as_metric_label(), "below_floor");
        let Error::TcbPolicy(msg) = err else {
            unreachable!()
        };
        assert!(msg.contains("bootloader"), "got: {msg}");
    }

    #[test]
    fn policy_rejects_below_snp_and_microcode_floor() {
        let policy = SevSnpTcbPolicy::genoa();
        let reported = svns(0x07, 0x00, 0x05, 0x10);
        let committed = reported;
        let (_, result) = policy.evaluate(
            &report_with_tcb(reported, committed),
            SevSnpGeneration::Genoa,
        );
        let Error::TcbPolicy(msg) = result.unwrap_err() else {
            panic!("expected TcbPolicy error");
        };
        assert!(msg.contains("snp"), "got: {msg}");
        assert!(msg.contains("microcode"), "got: {msg}");
    }

    #[test]
    fn policy_detects_rollback_even_when_above_floor() {
        let policy = SevSnpTcbPolicy::genoa();
        // reported is above the floor on every component, but the
        // firmware has committed to a higher snp SVN. This is the case
        // a malicious hypervisor SET_TCB_VERSION call would produce.
        let reported = svns(0x10, 0x05, 0x10, 0x80);
        let committed = svns(0x10, 0x05, 0x14, 0x80);
        let (obs, result) = policy.evaluate(
            &report_with_tcb(reported, committed),
            SevSnpGeneration::Genoa,
        );
        let Error::TcbPolicy(msg) = result.unwrap_err() else {
            panic!("expected TcbPolicy error");
        };
        assert_eq!(obs.as_metric_label(), "rollback_detected");
        assert!(msg.contains("rollback"), "got: {msg}");
    }

    #[test]
    fn policy_rollback_takes_precedence_over_below_floor() {
        // Both rollback and below-floor: the more severe condition wins
        // and the bucket label reflects rollback.
        let policy = SevSnpTcbPolicy::genoa();
        let reported = svns(0x05, 0x00, 0x05, 0x10);
        let committed = svns(0x07, 0x00, 0x0E, 0x48);
        let (obs, result) = policy.evaluate(
            &report_with_tcb(reported, committed),
            SevSnpGeneration::Genoa,
        );
        assert!(result.is_err());
        assert_eq!(obs.as_metric_label(), "rollback_detected");
    }

    #[test]
    fn default_floors_are_each_generations_own() {
        let floors = SevSnpTcbFloors::default();
        assert_eq!(floors.genoa, SevSnpTcbPolicy::genoa());
        assert_eq!(floors.turin, SevSnpTcbPolicy::turin());
        assert_eq!(
            floors.for_generation(SevSnpGeneration::Turin),
            &SevSnpTcbPolicy::turin()
        );
        assert_eq!(
            floors.for_generation(SevSnpGeneration::Genoa),
            &SevSnpTcbPolicy::genoa()
        );
    }

    /// Minimal parseable version-3 (Genoa) report byte buffer with a
    /// production-shaped guest policy, mutated per test.
    pub(crate) fn v3_report_bytes(mutate: impl FnOnce(&mut [u8; 1184])) -> Vec<u8> {
        let mut b = [0u8; 1184];
        b[0..4].copy_from_slice(&3u32.to_le_bytes()); // version 3
        b[0x08..0x10].copy_from_slice(&0x30000u64.to_le_bytes()); // policy: SMT + reserved-1 bit
        b[0x34..0x38].copy_from_slice(&1u32.to_le_bytes()); // sig_algo: ECDSA P-384
        b[0x188] = 0x19; // CPUID family: Genoa
        b[0x189] = 0x11; // CPUID model
        b[0x18A] = 0x01; // CPUID stepping
        mutate(&mut b);
        b.to_vec()
    }

    #[test]
    fn parse_rejects_wrong_report_length() {
        let ok = v3_report_bytes(|_| {});
        assert!(parse_report(&ok).is_ok());
        let mut long = ok.clone();
        long.push(0);
        assert!(parse_report(&long).is_err(), "trailing byte must fail");
        assert!(parse_report(&ok[..1183]).is_err(), "short report must fail");
    }

    #[test]
    fn hygiene_accepts_production_shaped_report() {
        let report = parse_report(&v3_report_bytes(|_| {})).unwrap();
        check_report_hygiene(&report).expect("clean v3 report must pass");
    }

    #[test]
    fn hygiene_rejects_report_version_below_3() {
        // A version-2 report needs the chip_id heuristic for generation
        // detection instead of the CPUID fields.
        let bytes = v3_report_bytes(|b| {
            b[0..4].copy_from_slice(&2u32.to_le_bytes());
            b[0x188] = 0;
            b[0x189] = 0;
            b[0x18A] = 0;
            b[0x1A0 + 8] = 0x01; // chip_id byte 8+ nonzero → Genoa detection
        });
        let report = parse_report(&bytes).unwrap();
        let err = check_report_hygiene(&report).unwrap_err();
        assert!(err.to_string().contains("version"), "got: {err}");
    }

    #[test]
    fn hygiene_rejects_debug_allowed_policy() {
        let bytes = v3_report_bytes(|b| b[0x0A] |= 0x08); // policy bit 19: DEBUG
        let report = parse_report(&bytes).unwrap();
        let err = check_report_hygiene(&report).unwrap_err();
        assert!(err.to_string().contains("DEBUG"), "got: {err}");
    }

    #[test]
    fn hygiene_rejects_migration_agent_policy() {
        let bytes = v3_report_bytes(|b| b[0x0A] |= 0x04); // policy bit 18: MIGRATE_MA
        let report = parse_report(&bytes).unwrap();
        let err = check_report_hygiene(&report).unwrap_err();
        assert!(err.to_string().contains("migration"), "got: {err}");
    }

    #[test]
    fn hygiene_rejects_masked_chip_key_and_non_vcek_signer() {
        let masked = v3_report_bytes(|b| b[0x48] |= 0x02); // MASK_CHIP_KEY
        let err = check_report_hygiene(&parse_report(&masked).unwrap()).unwrap_err();
        assert!(err.to_string().contains("MASK_CHIP_KEY"), "got: {err}");

        let vlek = v3_report_bytes(|b| b[0x48] |= 0x04); // SIGNING_KEY = 1 (VLEK)
        let err = check_report_hygiene(&parse_report(&vlek).unwrap()).unwrap_err();
        assert!(err.to_string().contains("SIGNING_KEY"), "got: {err}");
    }

    #[test]
    fn hygiene_rejects_id_block_launches() {
        for mutate in [
            (|b: &mut [u8; 1184]| b[0x48] |= 0x01) as fn(&mut [u8; 1184]), // AUTHOR_KEY_EN
            |b| b[0xE0] = 0xAA,                                            // ID_KEY_DIGEST
            |b| b[0x110] = 0xBB,                                           // AUTHOR_KEY_DIGEST
        ] {
            let bytes = v3_report_bytes(mutate);
            let err = check_report_hygiene(&parse_report(&bytes).unwrap()).unwrap_err();
            assert!(err.to_string().contains("ID-block"), "got: {err}");
        }
    }

    #[test]
    fn hygiene_rejects_id_block_fields_and_lower_vmpls() {
        for (mutate, needle) in [
            (
                (|b: &mut [u8; 1184]| b[0x04] = 1) as fn(&mut [u8; 1184]),
                "GUEST_SVN",
            ),
            (|b| b[0x10] = 1, "FAMILY_ID"),
            (|b| b[0x20] = 1, "IMAGE_ID"),
            (|b| b[0x30] = 1, "VMPL 1"),
            (|b| b[0x30] = 3, "VMPL 3"),
        ] {
            let bytes = v3_report_bytes(mutate);
            let err = check_report_hygiene(&parse_report(&bytes).unwrap()).unwrap_err();
            assert!(err.to_string().contains(needle), "{needle}: {err}");
        }
    }

    #[test]
    fn the_raw_policy_is_read_as_the_report_carries_it() {
        // The reserved bit clear: the `sev` crate's conversion would set it.
        let bytes = v3_report_bytes(|b| b[0x08..0x10].copy_from_slice(&0x10133u64.to_le_bytes()));
        assert_eq!(raw_policy(&bytes).unwrap(), 0x10133);
        assert!(raw_policy(&bytes[..12]).is_err());
    }

    #[test]
    fn the_guest_policy_must_equal_the_pin_but_for_a_higher_abi_minimum() {
        let pinned = 0x30133;
        check_guest_policy(0x30133, pinned).expect("equal");
        check_guest_policy(0x30134, pinned).expect("newer ABI minor");
        check_guest_policy(0x30200, pinned).expect("newer ABI major");
        for (reported, needle) in [
            (0xb0133, "outside the firmware ABI"),  // DEBUG
            (0x70133, "outside the firmware ABI"),  // MIGRATE_MA
            (0x230133, "outside the firmware ABI"), // CXL_ALLOW
            (0x130133, "outside the firmware ABI"), // SINGLE_SOCKET
            (0x20133, "outside the firmware ABI"),  // SMT cleared
            (0x10133, "outside the firmware ABI"),  // reserved bit cleared
            (0x30132, "older firmware"),
            (0x300ff, "older firmware"),
            (0x30000, "older firmware"),
        ] {
            let err = check_guest_policy(reported, pinned)
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{reported:#x}: {err}");
        }
    }

    #[test]
    fn policy_can_tighten_individual_components() {
        let mut policy = SevSnpTcbPolicy::genoa();
        policy.min_snp = 0x14;
        let reported = svns(0x07, 0x00, 0x10, 0x48); // above old floor, below new
        let (obs, result) = policy.evaluate(
            &report_with_tcb(reported, reported),
            SevSnpGeneration::Genoa,
        );
        assert!(result.is_err());
        assert_eq!(obs.as_metric_label(), "below_floor");
    }

    // --- VCEK binding -----------------------------------------------------
    //
    // The binding reads only the VCEK's extensions, so a synthetic VCEK is
    // the built-in Genoa ASK with its extensions replaced. Nothing here is
    // signed; `verify_report` checks the signature before the binding.

    const HWID: [u8; 64] = [0xA5; 64];

    fn kds_extension(
        oid: der::asn1::ObjectIdentifier,
        value: Vec<u8>,
    ) -> x509_cert::ext::Extension {
        x509_cert::ext::Extension {
            extn_id: oid,
            critical: false,
            extn_value: der::asn1::OctetString::new(value).unwrap(),
        }
    }

    fn spl_extension(oid: der::asn1::ObjectIdentifier, spl: u8) -> x509_cert::ext::Extension {
        kds_extension(oid, spl.to_der().unwrap())
    }

    /// A VCEK endorsed for `tcb` on chip `HWID`, with the hwID encoded the
    /// way AMD's KDS writes it: 64 raw bytes, no OCTET STRING tag.
    fn kds_extensions(tcb: SevSnpTcbSvns) -> Vec<x509_cert::ext::Extension> {
        vec![
            spl_extension(OID_BL_SPL, tcb.bootloader),
            spl_extension(OID_TEE_SPL, tcb.tee),
            spl_extension(OID_SNP_SPL, tcb.snp),
            spl_extension(OID_UCODE_SPL, tcb.microcode),
            kds_extension(OID_HWID, HWID.to_vec()),
        ]
    }

    fn vcek_with(extensions: Vec<x509_cert::ext::Extension>) -> Vec<u8> {
        let der = genoa::ask().unwrap().to_der().unwrap();
        let mut cert = x509_cert::Certificate::from_der(&der).unwrap();
        cert.tbs_certificate.extensions = Some(extensions);
        cert.to_der().unwrap()
    }

    fn report_signed_for(tcb: SevSnpTcbSvns, chip_id: [u8; 64]) -> AttestationReport {
        AttestationReport {
            chip_id,
            ..report_with_tcb(tcb, tcb)
        }
    }

    #[test]
    fn vcek_binding_accepts_the_tcb_and_chip_the_vcek_was_endorsed_for() {
        let tcb = svns(0x07, 0x00, 0x0E, 0x48);
        let report = report_signed_for(tcb, HWID);
        check_vcek_binding(
            &vcek_with(kds_extensions(tcb)),
            &report,
            SevSnpGeneration::Genoa,
        )
        .expect("raw hwID");

        // go-sev-guest also tolerates a tagged OCTET STRING hwID.
        let mut tagged = kds_extensions(tcb);
        tagged[4] = kds_extension(
            OID_HWID,
            der::asn1::OctetString::new(HWID.to_vec())
                .unwrap()
                .to_der()
                .unwrap(),
        );
        check_vcek_binding(&vcek_with(tagged), &report, SevSnpGeneration::Genoa)
            .expect("tagged hwID");
    }

    #[test]
    fn vcek_binding_refuses_a_report_claiming_a_tcb_its_vcek_was_not_endorsed_for() {
        // The leaked key's VCEK is for an old microcode; the report claims the
        // patched one, which every floor would otherwise accept.
        let endorsed = svns(0x07, 0x00, 0x0E, 0x0A);
        let claimed = svns(0x07, 0x00, 0x0E, 0x54);
        let vcek = vcek_with(kds_extensions(endorsed));
        let err = check_vcek_binding(
            &vcek,
            &report_signed_for(claimed, HWID),
            SevSnpGeneration::Genoa,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("not the TCB its VCEK was endorsed for"),
            "{err}"
        );
        // Each component is compared.
        let bl = svns(0x05, 0x00, 0x0E, 0x0A);
        assert!(
            check_vcek_binding(&vcek, &report_signed_for(bl, HWID), SevSnpGeneration::Genoa)
                .is_err()
        );
    }

    #[test]
    fn vcek_binding_refuses_another_chips_vcek() {
        let tcb = svns(0x07, 0x00, 0x0E, 0x48);
        let vcek = vcek_with(kds_extensions(tcb));
        let err = check_vcek_binding(
            &vcek,
            &report_signed_for(tcb, [0x5A; 64]),
            SevSnpGeneration::Genoa,
        )
        .unwrap_err();
        assert!(err.to_string().contains("CHIP_ID"), "{err}");
    }

    #[test]
    fn vcek_binding_tolerates_a_masked_chip_id() {
        // MASK_CHIP_ID zeroes CHIP_ID; there is nothing to compare, as in
        // go-sev-guest. The TCB binding still holds.
        let tcb = svns(0x07, 0x00, 0x0E, 0x48);
        let vcek = vcek_with(kds_extensions(tcb));
        check_vcek_binding(
            &vcek,
            &report_signed_for(tcb, [0; 64]),
            SevSnpGeneration::Genoa,
        )
        .expect("masked chip id");
        let claimed = svns(0x07, 0x00, 0x0E, 0x54);
        assert!(
            check_vcek_binding(
                &vcek,
                &report_signed_for(claimed, [0; 64]),
                SevSnpGeneration::Genoa
            )
            .is_err()
        );
    }

    #[test]
    fn vcek_binding_refuses_missing_duplicate_or_malformed_extensions() {
        let tcb = svns(0x07, 0x00, 0x0E, 0x48);
        let report = report_signed_for(tcb, HWID);

        let mut no_hwid = kds_extensions(tcb);
        no_hwid.pop();
        let err =
            check_vcek_binding(&vcek_with(no_hwid), &report, SevSnpGeneration::Genoa).unwrap_err();
        assert!(err.to_string().contains("no hwID extension"), "{err}");

        let mut twice = kds_extensions(tcb);
        twice.push(spl_extension(OID_UCODE_SPL, tcb.microcode));
        let err =
            check_vcek_binding(&vcek_with(twice), &report, SevSnpGeneration::Genoa).unwrap_err();
        assert!(err.to_string().contains("duplicate ucodeSPL"), "{err}");

        let mut wide = kds_extensions(tcb);
        wide[0] = kds_extension(OID_BL_SPL, 0x0107u16.to_der().unwrap());
        let err =
            check_vcek_binding(&vcek_with(wide), &report, SevSnpGeneration::Genoa).unwrap_err();
        assert!(err.to_string().contains("not a DER INTEGER 0-255"), "{err}");

        let mut short_hwid = kds_extensions(tcb);
        short_hwid[4] = kds_extension(OID_HWID, vec![0xA5; 32]);
        let err = check_vcek_binding(&vcek_with(short_hwid), &report, SevSnpGeneration::Genoa)
            .unwrap_err();
        assert!(err.to_string().contains("not 64 bytes"), "{err}");

        // The ASK itself carries no KDS extensions at all.
        let ask = genoa::ask().unwrap().to_der().unwrap();
        assert!(check_vcek_binding(&ask, &report, SevSnpGeneration::Genoa).is_err());
    }

    // --- Generations ------------------------------------------------------

    /// A version-3 Turin report (CPUID family 1Ah, model 02h), otherwise as
    /// [`v3_report_bytes`].
    pub(crate) fn turin_v3_report_bytes(mutate: impl FnOnce(&mut [u8; 1184])) -> Vec<u8> {
        v3_report_bytes(|b| {
            b[0x188] = 0x1A;
            b[0x189] = 0x02;
            b[0x18A] = 0x01;
            mutate(b);
        })
    }

    /// Write a TCB_VERSION in Turin's layout: `[fmc, bl, tee, snp, 0, 0, 0, ucode]`.
    fn write_turin_tcb(b: &mut [u8; 1184], offset: usize, tcb: SevSnpTcbSvns) {
        b[offset..offset + 8].copy_from_slice(&[
            tcb.fmc.unwrap(),
            tcb.bootloader,
            tcb.tee,
            tcb.snp,
            0,
            0,
            0,
            tcb.microcode,
        ]);
    }

    fn turin_svns(fmc: u8, bootloader: u8, tee: u8, snp: u8, microcode: u8) -> SevSnpTcbSvns {
        SevSnpTcbSvns {
            bootloader,
            tee,
            snp,
            microcode,
            fmc: Some(fmc),
        }
    }

    /// A parsed Turin report with these reported and committed TCBs.
    fn turin_report(reported: SevSnpTcbSvns, committed: SevSnpTcbSvns) -> AttestationReport {
        parse_report(&turin_v3_report_bytes(|b| {
            write_turin_tcb(b, 0x180, reported);
            write_turin_tcb(b, 0x1E0, committed);
        }))
        .unwrap()
    }

    #[test]
    fn the_generation_is_the_one_the_reports_cpuid_names() {
        let genoa = parse_report(&v3_report_bytes(|_| {})).unwrap();
        assert_eq!(
            SevSnpGeneration::of_report(&genoa).unwrap(),
            SevSnpGeneration::Genoa
        );

        // Turin's TCB layout is read: FMC first, microcode last.
        let tcb = turin_svns(1, 2, 3, 4, 82);
        let turin = turin_report(tcb, tcb);
        assert_eq!(
            SevSnpGeneration::of_report(&turin).unwrap(),
            SevSnpGeneration::Turin
        );
        assert_eq!(SevSnpTcbSvns::from(turin.reported_tcb), tcb);
        assert_eq!(SevSnpTcbSvns::from(turin.committed_tcb), tcb);

        // Generations with no pinned chain.
        for (family, model, name) in [(0x19, 0x01, "Milan"), (0x1A, 0x50, "Venice")] {
            let report = parse_report(&v3_report_bytes(|b| {
                b[0x188] = family;
                b[0x189] = model;
            }))
            .unwrap();
            let err = SevSnpGeneration::of_report(&report).unwrap_err();
            assert!(
                err.to_string().contains("neither Genoa nor Turin"),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn a_report_naming_no_generation_or_contradicting_its_own_is_refused() {
        // Without CPUID fields, as a version-2 report parses.
        let report = AttestationReport {
            version: 2,
            ..Default::default()
        };
        let err = SevSnpGeneration::of_report(&report).unwrap_err();
        assert!(err.to_string().contains("no CPUID"), "{err}");

        // Turin CPUID with a TCB read in Genoa's layout, and the reverse.
        let tcb = turin_svns(1, 1, 1, 4, 82);
        let mut turin = turin_report(tcb, tcb);
        turin.committed_tcb.fmc = None;
        let err = SevSnpGeneration::of_report(&turin).unwrap_err();
        assert!(err.to_string().contains("layout"), "{err}");
        let mut genoa = parse_report(&v3_report_bytes(|_| {})).unwrap();
        genoa.reported_tcb.fmc = Some(1);
        let err = SevSnpGeneration::of_report(&genoa).unwrap_err();
        assert!(err.to_string().contains("layout"), "{err}");
    }

    #[test]
    fn each_generation_resolves_its_own_chain_and_an_override_replaces_both() {
        let genoa = resolve_chain_certs_der(SevSnpGeneration::Genoa, None, None).unwrap();
        let turin = resolve_chain_certs_der(SevSnpGeneration::Turin, None, None).unwrap();
        assert_ne!(genoa.0, turin.0);
        assert_ne!(genoa.1, turin.1);
        for generation in [SevSnpGeneration::Genoa, SevSnpGeneration::Turin] {
            let overridden =
                resolve_chain_certs_der(generation, Some(b"ark"), Some(b"ask")).unwrap();
            assert_eq!(
                overridden,
                (b"ark".to_vec(), b"ask".to_vec()),
                "{generation}"
            );
        }
    }

    /// A Turin VCEK endorsed for `tcb` on the 8-byte chip `TURIN_HWID`, the
    /// hwID raw as the KDS writes it.
    const TURIN_HWID: [u8; 8] = [0x2A, 0x24, 0xDC, 0x0D, 0x1F, 0xA8, 0x66, 0x5F];

    fn turin_kds_extensions(tcb: SevSnpTcbSvns) -> Vec<x509_cert::ext::Extension> {
        vec![
            spl_extension(OID_FMC_SPL, tcb.fmc.unwrap()),
            spl_extension(OID_BL_SPL, tcb.bootloader),
            spl_extension(OID_TEE_SPL, tcb.tee),
            spl_extension(OID_SNP_SPL, tcb.snp),
            spl_extension(OID_UCODE_SPL, tcb.microcode),
            kds_extension(OID_HWID, TURIN_HWID.to_vec()),
        ]
    }

    /// A Turin CHIP_ID: the 8-byte hwID, zero-padded to 64.
    fn turin_chip_id(hwid: [u8; 8]) -> [u8; 64] {
        let mut chip_id = [0u8; 64];
        chip_id[..8].copy_from_slice(&hwid);
        chip_id
    }

    fn turin_report_signed_for(tcb: SevSnpTcbSvns, chip_id: [u8; 64]) -> AttestationReport {
        AttestationReport {
            chip_id,
            ..turin_report(tcb, tcb)
        }
    }

    const TURIN: SevSnpGeneration = SevSnpGeneration::Turin;

    #[test]
    fn turin_vcek_binding_accepts_the_tcb_with_fmc_and_the_8_byte_chip() {
        let tcb = turin_svns(1, 1, 1, 4, 82);
        let report = turin_report_signed_for(tcb, turin_chip_id(TURIN_HWID));
        check_vcek_binding(&vcek_with(turin_kds_extensions(tcb)), &report, TURIN)
            .expect("raw hwID");

        let mut tagged = turin_kds_extensions(tcb);
        tagged[5] = kds_extension(
            OID_HWID,
            der::asn1::OctetString::new(TURIN_HWID.to_vec())
                .unwrap()
                .to_der()
                .unwrap(),
        );
        check_vcek_binding(&vcek_with(tagged), &report, TURIN).expect("tagged hwID");
    }

    #[test]
    fn turin_vcek_binding_refuses_a_report_claiming_an_fmc_its_vcek_was_not_endorsed_for() {
        let endorsed = turin_svns(1, 1, 1, 4, 82);
        let vcek = vcek_with(turin_kds_extensions(endorsed));
        let claimed = turin_svns(2, 1, 1, 4, 82);
        let err = check_vcek_binding(
            &vcek,
            &turin_report_signed_for(claimed, turin_chip_id(TURIN_HWID)),
            TURIN,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("not the TCB its VCEK was endorsed for"),
            "{err}"
        );
        // The other four components are still compared.
        let claimed = turin_svns(1, 1, 1, 4, 90);
        assert!(
            check_vcek_binding(
                &vcek,
                &turin_report_signed_for(claimed, turin_chip_id(TURIN_HWID)),
                TURIN
            )
            .is_err()
        );
    }

    #[test]
    fn the_fmc_extension_is_required_on_turin_and_refused_on_genoa() {
        let tcb = turin_svns(1, 1, 1, 4, 82);
        let report = turin_report_signed_for(tcb, turin_chip_id(TURIN_HWID));
        let mut no_fmc = turin_kds_extensions(tcb);
        no_fmc.remove(0);
        let err = check_vcek_binding(&vcek_with(no_fmc), &report, TURIN).unwrap_err();
        assert!(err.to_string().contains("no fmcSPL extension"), "{err}");

        let mut twice = turin_kds_extensions(tcb);
        twice.push(spl_extension(OID_FMC_SPL, 1));
        let err = check_vcek_binding(&vcek_with(twice), &report, TURIN).unwrap_err();
        assert!(err.to_string().contains("duplicate fmcSPL"), "{err}");

        let genoa_tcb = svns(0x07, 0x00, 0x0E, 0x48);
        let mut genoa_with_fmc = kds_extensions(genoa_tcb);
        genoa_with_fmc.push(spl_extension(OID_FMC_SPL, 1));
        let err = check_vcek_binding(
            &vcek_with(genoa_with_fmc),
            &report_signed_for(genoa_tcb, HWID),
            SevSnpGeneration::Genoa,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Genoa VCEK certificate carries an fmcSPL"),
            "{err}"
        );
    }

    #[test]
    fn the_hwid_length_is_the_generations() {
        // A Genoa-length hwID on a Turin VCEK, and a Turin-length one on a
        // Genoa VCEK, are each refused.
        let tcb = turin_svns(1, 1, 1, 4, 82);
        let mut wide = turin_kds_extensions(tcb);
        wide[5] = kds_extension(OID_HWID, turin_chip_id(TURIN_HWID).to_vec());
        let err = check_vcek_binding(
            &vcek_with(wide),
            &turin_report_signed_for(tcb, turin_chip_id(TURIN_HWID)),
            TURIN,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Turin VCEK hwID extension is not 8 bytes"),
            "{err}"
        );

        let genoa_tcb = svns(0x07, 0x00, 0x0E, 0x48);
        let mut narrow = kds_extensions(genoa_tcb);
        narrow[4] = kds_extension(OID_HWID, HWID[..8].to_vec());
        let err = check_vcek_binding(
            &vcek_with(narrow),
            &report_signed_for(genoa_tcb, HWID),
            SevSnpGeneration::Genoa,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Genoa VCEK hwID extension is not 64 bytes"),
            "{err}"
        );
    }

    #[test]
    fn every_byte_of_a_turin_chip_id_is_compared() {
        let tcb = turin_svns(1, 1, 1, 4, 82);
        let vcek = vcek_with(turin_kds_extensions(tcb));

        // Another chip.
        let mut other = TURIN_HWID;
        other[7] ^= 1;
        let err = check_vcek_binding(
            &vcek,
            &turin_report_signed_for(tcb, turin_chip_id(other)),
            TURIN,
        )
        .unwrap_err();
        assert!(err.to_string().contains("CHIP_ID"), "{err}");

        // The VCEK's chip, but a CHIP_ID that does not end in zeros.
        let mut padded = turin_chip_id(TURIN_HWID);
        padded[63] = 1;
        let err =
            check_vcek_binding(&vcek, &turin_report_signed_for(tcb, padded), TURIN).unwrap_err();
        assert!(err.to_string().contains("CHIP_ID"), "{err}");

        // Only the leading eight bytes zero is not a masked identity.
        let mut tail_only = [0u8; 64];
        tail_only[8] = 1;
        assert!(
            check_vcek_binding(&vcek, &turin_report_signed_for(tcb, tail_only), TURIN).is_err()
        );

        // A wholly zero CHIP_ID is masked, as on Genoa, and the TCB binding
        // still holds.
        check_vcek_binding(&vcek, &turin_report_signed_for(tcb, [0; 64]), TURIN)
            .expect("masked chip id");
        let claimed = turin_svns(2, 1, 1, 4, 82);
        assert!(
            check_vcek_binding(&vcek, &turin_report_signed_for(claimed, [0; 64]), TURIN).is_err()
        );
    }

    #[test]
    fn turin_reports_are_held_to_turins_floor_including_fmc() {
        let floors = SevSnpTcbFloors::default();
        let at_floor = turin_svns(1, 1, 1, 4, 82);
        let (obs, result) = floors.evaluate(&turin_report(at_floor, at_floor), TURIN);
        result.expect("at Turin's floor");
        assert_eq!(obs.generation, TURIN);
        assert_eq!(obs.as_metric_label(), "meets_floor");

        for (short, needle) in [
            (turin_svns(0, 1, 1, 4, 82), "fmc"),
            (turin_svns(1, 0, 1, 4, 82), "bootloader"),
            (turin_svns(1, 1, 0, 4, 82), "tee"),
            (turin_svns(1, 1, 1, 3, 82), "snp"),
            (turin_svns(1, 1, 1, 4, 81), "microcode"),
        ] {
            let (obs, result) = floors.evaluate(&turin_report(short, short), TURIN);
            let err = result.unwrap_err().to_string();
            assert!(err.contains(needle), "{needle}: {err}");
            assert_eq!(obs.as_metric_label(), "below_floor");
        }

        // Each generation's numbers are its own: Turin's floor is far below
        // Genoa's bootloader floor, which would refuse every Turin report.
        let (_, result) = floors
            .genoa
            .evaluate(&turin_report(at_floor, at_floor), TURIN);
        assert!(result.unwrap_err().to_string().contains("bootloader"));
        let genoa_at_floor = svns(0x07, 0x00, 0x0E, 0x48);
        let (obs, result) = floors.evaluate(
            &report_with_tcb(genoa_at_floor, genoa_at_floor),
            SevSnpGeneration::Genoa,
        );
        result.expect("at Genoa's floor");
        assert_eq!(obs.generation, SevSnpGeneration::Genoa);
    }

    #[test]
    fn an_fmc_below_the_committed_fmc_is_a_rollback() {
        let floors = SevSnpTcbFloors::default();
        let reported = turin_svns(1, 2, 2, 5, 90);
        let committed = turin_svns(2, 2, 2, 5, 90);
        let (obs, result) = floors.evaluate(&turin_report(reported, committed), TURIN);
        assert!(result.unwrap_err().to_string().contains("rollback"));
        assert_eq!(obs.as_metric_label(), "rollback_detected");
    }
}
