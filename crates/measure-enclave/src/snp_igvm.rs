//! The AMD SEV-SNP launch identity of a Tinfoil CVM in the IGVM launch model
//! (cvmimage v0.15 and later), computed from public inputs.
//!
//! In that model the guest is a single firmware-less IGVM image with no
//! measured boot after launch. A report from such a guest is identified by:
//!
//! - **MEASUREMENT**, the launch digest the AMD secure processor accumulates
//!   over every page the host adds before the guest runs, then the boot
//!   processor's save area (VMSA). Hosts launch with
//!   `KVM_SEV_SNP_MEASURE_BSP_ONLY`, so only the boot processor's VMSA is
//!   measured and the digest does not depend on the processor count; RAM size
//!   reaches the guest through an unmeasured parameter page. It is a function
//!   of the image alone, the same for every machine shape and every
//!   deployment of that image. [`launch_digest_from_igvm`] recomputes it.
//! - **HOST_DATA**, 32 bytes the host passes at launch, which the firmware
//!   copies into every report and which the guest reads back and checks
//!   against its config before it boots: the SHA-256 of the deployment's
//!   `tinfoil-config.yml` ([`host_data`]). This, not the digest, is what
//!   distinguishes one deployment of an image from another.
//! - **POLICY**, the guest policy the host passed at launch. It is not part of
//!   the digest; the image states the one it is built for in its IGVM
//!   initialization header, and [`release_pin`] reads it from there.
//!
//! [`release_pin`] ties the computation to a release: the release manifest is
//! pinned by SHA-256 (the config's inline `cvm-version` pin), the IGVM image
//! must hash to what that manifest records, and the recomputed digest must
//! equal the one the manifest states. A disagreement anywhere is an error
//! rather than a choice between values.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use igvm::{IgvmDirectiveHeader, IgvmFile, IgvmInitializationHeader, IgvmPlatformHeader};
use igvm_defs::{IgvmPageDataType, IgvmPlatformType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, Sha384};
use zerocopy::IntoBytes;

use crate::tdx_igvm::config_release;

const PAGE_SIZE: u64 = 4096;
/// A 2 MiB page directive covers this many 4 KiB pages.
const PAGES_PER_2MB: u64 = 512;

/// The guest-physical address KVM measures the boot processor's VMSA at.
pub const VMSA_GPA: u64 = 0xffff_ffff_f000;

/// `PAGE_INFO`, the structure `SNP_LAUNCH_UPDATE` folds into the launch
/// digest for every page (AMD SEV-SNP Firmware ABI, `SNP_LAUNCH_UPDATE`):
/// the digest so far (48 bytes), the page's contents digest (48), the
/// structure's length (`u16`), the page type, `IMI_PAGE`, three VMPL
/// permission bytes, reserved, and the guest-physical address (`u64`).
const PAGE_INFO_LEN: usize = 0x70;
const PAGE_INFO_CONTENTS: usize = 0x30;
const PAGE_INFO_LENGTH: usize = 0x60;
const PAGE_INFO_TYPE: usize = 0x62;
const PAGE_INFO_GPA: usize = 0x68;

/// `PAGE_TYPE` values. Only normal and VMSA pages have their contents
/// digested; the others contribute their type and address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum PageType {
    Normal = 0x1,
    Vmsa = 0x2,
    Unmeasured = 0x4,
    Secrets = 0x5,
    Cpuid = 0x6,
}

/// The launch digest accumulator.
struct LaunchDigest([u8; 48]);

impl LaunchDigest {
    fn update(&mut self, gpa: u64, page_type: PageType, contents: Option<&[u8]>) {
        let mut info = [0u8; PAGE_INFO_LEN];
        info[..PAGE_INFO_CONTENTS].copy_from_slice(&self.0);
        if let Some(page) = contents {
            info[PAGE_INFO_CONTENTS..PAGE_INFO_LENGTH].copy_from_slice(&Sha384::digest(page));
        }
        info[PAGE_INFO_LENGTH..PAGE_INFO_LENGTH + 2]
            .copy_from_slice(&(PAGE_INFO_LEN as u16).to_le_bytes());
        info[PAGE_INFO_TYPE] = page_type as u8;
        info[PAGE_INFO_GPA..PAGE_INFO_GPA + 8].copy_from_slice(&gpa.to_le_bytes());
        self.0 = Sha384::digest(info).into();
    }
}

/// What an SEV-SNP IGVM image fixes about the reports its launches produce.
#[derive(Debug, PartialEq, Eq)]
pub struct SnpImage {
    /// The launch digest.
    pub measurement: [u8; 48],
    /// The guest policy the image's initialization header asks for.
    pub guest_policy: u64,
}

/// Recompute the launch digest and read the guest policy of an SEV-SNP IGVM
/// image.
///
/// Directives are applied in file order, as a loader imports them: an
/// inserted parameter area is an unmeasured page at its address, page data is
/// a normal page (a page with no data is a page of zeros), a secrets or CPUID
/// page is measured by type and address only, and a 2 MiB zero page is the
/// 512 4 KiB zero pages it holds, in ascending order (cvmimage's image
/// compiler emits each 2 MiB-aligned run of zero pages that way). The boot
/// processor's VMSA comes last, at [`VMSA_GPA`]. Directives that only
/// describe parameters or memory the host supplies after measurement
/// contribute nothing. Anything this computation does not model is refused
/// rather than guessed at: a page marked unmeasured or shared, another page
/// data type, a 2 MiB page carrying data or misaligned, a page of the wrong
/// length, a parameter area inserted after page data, a VMSA anywhere but
/// [`VMSA_GPA`] or for any processor but the boot processor (hosts measure
/// that one only), and an ID block, which the verifier refuses at launch.
pub fn launch_digest_from_igvm(bytes: &[u8]) -> Result<SnpImage> {
    let file = IgvmFile::new_from_binary(bytes, None).context("parsing IGVM image")?;
    let snp_mask = match file.platforms() {
        [IgvmPlatformHeader::SupportedPlatform(platform)]
            if platform.platform_type == IgvmPlatformType::SEV_SNP =>
        {
            platform.compatibility_mask
        }
        other => bail!(
            "IGVM image must declare exactly one platform, SEV-SNP; it declares {}",
            other.len()
        ),
    };

    let mut policies = file.initializations().iter().filter_map(|init| match init {
        IgvmInitializationHeader::GuestPolicy {
            policy,
            compatibility_mask,
        } if compatibility_mask & snp_mask != 0 => Some(*policy),
        _ => None,
    });
    let guest_policy = policies
        .next()
        .context("IGVM image states no SEV-SNP guest policy")?;
    ensure!(
        policies.next().is_none(),
        "IGVM image states more than one SEV-SNP guest policy"
    );

    let mut parameter_areas = HashMap::new();
    let mut digest = LaunchDigest([0; 48]);
    let mut page_data_seen = false;
    let mut vmsa: Option<Vec<u8>> = None;
    let zero_page = [0u8; PAGE_SIZE as usize];
    for directive in file.directives() {
        match directive {
            IgvmDirectiveHeader::PageData {
                gpa,
                compatibility_mask,
                flags,
                data_type,
                data,
            } => {
                if compatibility_mask & snp_mask == 0 {
                    continue;
                }
                ensure!(
                    vmsa.is_none(),
                    "page at {gpa:#x} follows the VMSA; the save area is measured last"
                );
                ensure!(
                    !flags.unmeasured() && !flags.shared(),
                    "page at {gpa:#x} is marked unmeasured or shared"
                );
                page_data_seen = true;
                if flags.is_2mb_page() {
                    ensure!(
                        *data_type == IgvmPageDataType::NORMAL && data.is_empty(),
                        "2 MiB page at {gpa:#x} carries data; only a 2 MiB zero page is supported"
                    );
                    ensure!(
                        gpa.is_multiple_of(PAGES_PER_2MB * PAGE_SIZE),
                        "2 MiB page at {gpa:#x} is not 2 MiB-aligned"
                    );
                    for page in 0..PAGES_PER_2MB {
                        digest.update(gpa + page * PAGE_SIZE, PageType::Normal, Some(&zero_page));
                    }
                    continue;
                }
                ensure!(
                    data.is_empty() || data.len() == PAGE_SIZE as usize,
                    "page at {gpa:#x} carries {} bytes",
                    data.len()
                );
                match *data_type {
                    IgvmPageDataType::NORMAL => {
                        let contents: &[u8] = if data.is_empty() { &zero_page } else { data };
                        digest.update(*gpa, PageType::Normal, Some(contents));
                    }
                    IgvmPageDataType::SECRETS => digest.update(*gpa, PageType::Secrets, None),
                    IgvmPageDataType::CPUID_DATA => digest.update(*gpa, PageType::Cpuid, None),
                    other => bail!("page at {gpa:#x} has unsupported data type {other:?}"),
                }
            }
            IgvmDirectiveHeader::ParameterArea {
                number_of_bytes,
                parameter_area_index,
                initial_data,
            } => {
                ensure!(
                    initial_data.is_empty(),
                    "parameter area {parameter_area_index} carries initial data"
                );
                parameter_areas.insert(*parameter_area_index, *number_of_bytes);
            }
            IgvmDirectiveHeader::ParameterInsert(insert) => {
                if insert.compatibility_mask & snp_mask == 0 {
                    continue;
                }
                ensure!(
                    !page_data_seen && vmsa.is_none(),
                    "parameter area {} is inserted after page data",
                    insert.parameter_area_index
                );
                let size = parameter_areas
                    .get(&insert.parameter_area_index)
                    .with_context(|| {
                        format!(
                            "parameter area {} inserted before it is declared",
                            insert.parameter_area_index
                        )
                    })?;
                for page in 0..size.div_ceil(PAGE_SIZE) {
                    digest.update(insert.gpa + page * PAGE_SIZE, PageType::Unmeasured, None);
                }
            }
            IgvmDirectiveHeader::SnpVpContext {
                gpa,
                compatibility_mask,
                vp_index,
                vmsa: context,
            } => {
                if compatibility_mask & snp_mask == 0 {
                    continue;
                }
                ensure!(
                    *vp_index == 0,
                    "IGVM image carries a VMSA for processor {vp_index}; hosts measure the \
                     boot processor's only"
                );
                ensure!(
                    vmsa.is_none(),
                    "IGVM image carries two boot processor VMSAs"
                );
                ensure!(
                    *gpa == VMSA_GPA,
                    "VMSA at {gpa:#x}; KVM measures the save area at {VMSA_GPA:#x}"
                );
                let page = context.as_bytes();
                ensure!(
                    page.len() == PAGE_SIZE as usize,
                    "VMSA is {} bytes, not one page",
                    page.len()
                );
                vmsa = Some(page.to_vec());
            }
            IgvmDirectiveHeader::SnpIdBlock { .. } => bail!(
                "IGVM image carries an ID block; the verifier accepts only launches without one"
            ),
            IgvmDirectiveHeader::VpCount(_)
            | IgvmDirectiveHeader::EnvironmentInfo(_)
            | IgvmDirectiveHeader::Srat(_)
            | IgvmDirectiveHeader::Madt(_)
            | IgvmDirectiveHeader::Slit(_)
            | IgvmDirectiveHeader::Pptt(_)
            | IgvmDirectiveHeader::MmioRanges(_)
            | IgvmDirectiveHeader::MemoryMap(_)
            | IgvmDirectiveHeader::CommandLine(_)
            | IgvmDirectiveHeader::DeviceTree(_)
            | IgvmDirectiveHeader::RequiredMemory { .. }
            | IgvmDirectiveHeader::ErrorRange { .. } => {}
            other => bail!("IGVM directive is outside the SEV-SNP measurement model: {other:?}"),
        }
    }
    let vmsa = vmsa.context("IGVM image carries no boot processor VMSA")?;
    digest.update(VMSA_GPA, PageType::Vmsa, Some(&vmsa));
    Ok(SnpImage {
        measurement: digest.0,
        guest_policy,
    })
}

/// HOST_DATA for a deployment config: the SHA-256 of the exact config bytes.
pub fn host_data(config: &[u8]) -> [u8; 32] {
    Sha256::digest(config).into()
}

/// The canonical spelling of a guest policy: lowercase hex, `0x`-prefixed,
/// no leading zeros (the release manifests' spelling, and the verifier's).
pub fn guest_policy_hex(policy: u64) -> String {
    format!("{policy:#x}")
}

/// The fields of a cvmimage release manifest this tool reads.
#[derive(Deserialize)]
struct ReleaseManifest {
    version: String,
    igvm: ReleaseIgvm,
}

#[derive(Deserialize)]
struct ReleaseIgvm {
    /// Layout version of this `igvm` block. Only version 1 is understood.
    format_version: u32,
    /// SHA-256 of the SEV-SNP IGVM image.
    snp: String,
    snp_launch: SnpLaunch,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SnpLaunch {
    measurement: String,
}

/// The SEV-SNP launch identity of one deployment of one release: the
/// `releases/trust/server-enclave.json` record, less its envelope.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct SnpLaunchPin {
    /// The config's `cvm-version`, exactly as written: the release and the
    /// SHA-256 of its manifest.
    pub cvm_version: String,
    /// SHA-256 of the IGVM image the measurement was computed from.
    pub igvm_sha256: String,
    /// The pin itself, in `tinfoil_verifier::SevSnpPin`'s shape.
    pub sev_snp: SevSnpLaunch,
}

/// The report fields a deployment fixes, in `tinfoil_verifier::SevSnpPin`'s
/// serde shape.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct SevSnpLaunch {
    /// Hex-encoded launch digest.
    pub measurement: String,
    /// Hex-encoded HOST_DATA: the config's SHA-256.
    pub host_data: String,
    /// The guest policy the image asks for (see [`guest_policy_hex`]).
    pub guest_policy: String,
}

/// Compute a deployment's SEV-SNP launch identity and check it against the
/// release that published the image.
///
/// The config must pin its release manifest inline (`cvm-version:
/// X@sha256:<hex>`), so the config alone names the image it launches; that
/// pin must be the manifest's hash, the manifest must describe that release,
/// the image must be the one the manifest records, and the recomputed digest
/// must equal the one the manifest states. HOST_DATA is computed from the
/// config bytes.
pub fn release_pin(manifest: &[u8], igvm: &[u8], config: &[u8]) -> Result<SnpLaunchPin> {
    let selected = config_release(config)?;
    let pinned = selected.manifest_sha256.as_deref().context(
        "cvm-version must pin its release manifest inline, as VERSION@sha256:<64 lowercase hex>",
    )?;
    let actual = hex::encode(Sha256::digest(manifest));
    ensure!(
        actual == pinned,
        "config pins release manifest {pinned}, but the manifest supplied is {actual}"
    );
    let parsed: ReleaseManifest =
        serde_json::from_slice(manifest).context("parsing release manifest")?;
    ensure!(
        parsed.igvm.format_version == 1,
        "release manifest igvm.format_version {} is not supported",
        parsed.igvm.format_version
    );
    ensure!(
        parsed.version == format!("v{}", selected.version),
        "config selects cvm-version {}, but the release manifest is {}",
        selected.version,
        parsed.version
    );

    let igvm_sha256 = hex::encode(Sha256::digest(igvm));
    ensure!(
        igvm_sha256 == parsed.igvm.snp.to_ascii_lowercase(),
        "IGVM image SHA-256 is {igvm_sha256}, but release {} records {}",
        parsed.version,
        parsed.igvm.snp
    );
    let image = launch_digest_from_igvm(igvm)?;
    let measurement = hex::encode(image.measurement);
    ensure!(
        measurement == parsed.igvm.snp_launch.measurement.to_ascii_lowercase(),
        "recomputed SEV-SNP launch digest {measurement} disagrees with the digest release {} \
         states, {}",
        parsed.version,
        parsed.igvm.snp_launch.measurement
    );
    Ok(SnpLaunchPin {
        cvm_version: format!("{}@sha256:{pinned}", selected.version),
        igvm_sha256,
        sev_snp: SevSnpLaunch {
            measurement,
            host_data: hex::encode(host_data(config)),
            guest_policy: guest_policy_hex(image.guest_policy),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use igvm::IgvmRevision;
    use igvm::snp_defs::SevVmsa;
    use igvm_defs::{
        IGVM_VHS_PARAMETER, IGVM_VHS_PARAMETER_INSERT, IGVM_VHS_SUPPORTED_PLATFORM,
        IgvmPageDataFlags,
    };
    use zerocopy::FromZeros;

    const SNP_MASK: u32 = 0x1;
    const POLICY: u64 = 0x30133;

    fn image_with(policy: Option<u64>, directives: Vec<IgvmDirectiveHeader>) -> Vec<u8> {
        let file = IgvmFile::new(
            IgvmRevision::V1,
            vec![IgvmPlatformHeader::SupportedPlatform(
                IGVM_VHS_SUPPORTED_PLATFORM {
                    compatibility_mask: SNP_MASK,
                    highest_vtl: 0,
                    platform_type: IgvmPlatformType::SEV_SNP,
                    platform_version: 1,
                    shared_gpa_boundary: 0,
                },
            )],
            policy
                .into_iter()
                .map(|policy| IgvmInitializationHeader::GuestPolicy {
                    policy,
                    compatibility_mask: SNP_MASK,
                })
                .collect(),
            directives,
        )
        .unwrap();
        let mut out = Vec::new();
        file.serialize(&mut out).unwrap();
        out
    }

    fn image(directives: Vec<IgvmDirectiveHeader>) -> Vec<u8> {
        image_with(Some(POLICY), directives)
    }

    fn page(gpa: u64, fill: u8) -> IgvmDirectiveHeader {
        IgvmDirectiveHeader::PageData {
            gpa,
            compatibility_mask: SNP_MASK,
            flags: IgvmPageDataFlags::new(),
            data_type: IgvmPageDataType::NORMAL,
            data: vec![fill; PAGE_SIZE as usize],
        }
    }

    fn typed(gpa: u64, data_type: IgvmPageDataType) -> IgvmDirectiveHeader {
        IgvmDirectiveHeader::PageData {
            gpa,
            compatibility_mask: SNP_MASK,
            flags: IgvmPageDataFlags::new(),
            data_type,
            data: vec![],
        }
    }

    fn vmsa_with(gpa: u64, vp_index: u16, rip: u64) -> IgvmDirectiveHeader {
        let mut vmsa = SevVmsa::new_box_zeroed().unwrap();
        vmsa.rip = rip;
        IgvmDirectiveHeader::SnpVpContext {
            gpa,
            compatibility_mask: SNP_MASK,
            vp_index,
            vmsa,
        }
    }

    fn vmsa() -> IgvmDirectiveHeader {
        vmsa_with(VMSA_GPA, 0, 0x1000)
    }

    fn vmsa_page(rip: u64) -> Vec<u8> {
        let mut vmsa = SevVmsa::new_box_zeroed().unwrap();
        vmsa.rip = rip;
        vmsa.as_bytes().to_vec()
    }

    /// The digest written out longhand from the firmware's `PAGE_INFO`
    /// layout, as an independent check of [`LaunchDigest`].
    fn longhand(pages: &[(u64, u8, Option<Vec<u8>>)]) -> [u8; 48] {
        let mut digest = [0u8; 48];
        for (gpa, page_type, contents) in pages {
            let mut info = [0u8; 112];
            info[..48].copy_from_slice(&digest);
            if let Some(contents) = contents {
                info[48..96].copy_from_slice(&Sha384::digest(contents));
            }
            info[96] = 0x70;
            info[98] = *page_type;
            info[104..112].copy_from_slice(&gpa.to_le_bytes());
            digest = Sha384::digest(info).into();
        }
        digest
    }

    /// The reference vector cvmimage's compiler checks its digest code
    /// against (`measurement_matches_the_reference_implementation`, a known
    /// answer from `sev-snp-measure`): one normal page whose first byte is 1,
    /// a secrets page, a CPUID page, then an all-zero VMSA.
    #[test]
    fn the_digest_matches_the_sev_snp_measure_reference_vector() {
        let mut digest = LaunchDigest([0; 48]);
        let mut first = vec![0u8; PAGE_SIZE as usize];
        first[0] = 1;
        digest.update(0x1000, PageType::Normal, Some(&first));
        digest.update(0x2000, PageType::Secrets, None);
        digest.update(0x3000, PageType::Cpuid, None);
        digest.update(VMSA_GPA, PageType::Vmsa, Some(&[0u8; PAGE_SIZE as usize]));
        assert_eq!(
            hex::encode(digest.0),
            "64fba8d7f08e6c2b07f7a3fd610e2965a1683a3ca18ad66a73acc84e5cc2ebfb\
             1721d1bcfebf8752aeac62b6fd5f8ace"
        );
    }

    #[test]
    fn every_page_kind_is_folded_in_file_order_and_the_vmsa_last() {
        let bytes = image(vec![
            IgvmDirectiveHeader::ParameterArea {
                number_of_bytes: PAGE_SIZE,
                parameter_area_index: 0,
                initial_data: vec![],
            },
            IgvmDirectiveHeader::VpCount(IGVM_VHS_PARAMETER {
                parameter_area_index: 0,
                byte_offset: 0,
            }),
            IgvmDirectiveHeader::ParameterInsert(IGVM_VHS_PARAMETER_INSERT {
                gpa: 0x5000,
                compatibility_mask: SNP_MASK,
                parameter_area_index: 0,
            }),
            page(0x2000, 0xaa),
            typed(0x3000, IgvmPageDataType::NORMAL),
            typed(0x6000, IgvmPageDataType::CPUID_DATA),
            typed(0x7000, IgvmPageDataType::SECRETS),
            vmsa(),
        ]);
        let image = launch_digest_from_igvm(&bytes).unwrap();
        assert_eq!(image.guest_policy, POLICY);
        assert_eq!(
            image.measurement,
            longhand(&[
                (0x5000, 4, None),
                (0x2000, 1, Some(vec![0xaa; 4096])),
                (0x3000, 1, Some(vec![0; 4096])),
                (0x6000, 6, None),
                (0x7000, 5, None),
                (VMSA_GPA, 2, Some(vmsa_page(0x1000))),
            ])
        );
    }

    #[test]
    fn the_digest_depends_on_order_address_content_and_the_vmsa() {
        let base = launch_digest_from_igvm(&image(vec![page(0x2000, 1), page(0x3000, 2), vmsa()]))
            .unwrap()
            .measurement;
        for variant in [
            vec![page(0x3000, 2), page(0x2000, 1), vmsa()],
            vec![page(0x2000, 1), page(0x4000, 2), vmsa()],
            vec![page(0x2000, 1), page(0x3000, 3), vmsa()],
            vec![
                page(0x2000, 1),
                page(0x3000, 2),
                vmsa_with(VMSA_GPA, 0, 0x2000),
            ],
        ] {
            assert_ne!(
                launch_digest_from_igvm(&image(variant))
                    .unwrap()
                    .measurement,
                base
            );
        }
    }

    #[test]
    fn a_2mb_zero_page_is_its_512_zero_pages() {
        let base = 0x20_0000;
        let huge = launch_digest_from_igvm(&image(vec![
            page(0x1f_f000, 0xaa),
            IgvmDirectiveHeader::PageData {
                gpa: base,
                compatibility_mask: SNP_MASK,
                flags: IgvmPageDataFlags::new().with_is_2mb_page(true),
                data_type: IgvmPageDataType::NORMAL,
                data: vec![],
            },
            vmsa(),
        ]))
        .unwrap()
        .measurement;
        let mut small = vec![page(0x1f_f000, 0xaa)];
        small.extend(
            (0..PAGES_PER_2MB).map(|i| typed(base + i * PAGE_SIZE, IgvmPageDataType::NORMAL)),
        );
        small.push(vmsa());
        assert_eq!(
            huge,
            launch_digest_from_igvm(&image(small)).unwrap().measurement
        );
    }

    #[test]
    fn what_the_model_does_not_cover_is_refused() {
        let flagged = |flags: IgvmPageDataFlags| IgvmDirectiveHeader::PageData {
            gpa: 0x2000,
            compatibility_mask: SNP_MASK,
            flags,
            data_type: IgvmPageDataType::NORMAL,
            data: vec![1; PAGE_SIZE as usize],
        };
        let cases: Vec<(&str, Vec<u8>)> = vec![
            (
                "no SEV-SNP guest policy",
                image_with(None, vec![page(0x2000, 1), vmsa()]),
            ),
            ("no boot processor VMSA", image(vec![page(0x2000, 1)])),
            (
                "processor 1",
                image(vec![page(0x2000, 1), vmsa(), vmsa_with(VMSA_GPA, 1, 0)]),
            ),
            (
                "KVM measures",
                image(vec![page(0x2000, 1), vmsa_with(0x9000, 0, 0)]),
            ),
            (
                "follows the VMSA",
                image(vec![page(0x2000, 1), vmsa(), page(0x3000, 1)]),
            ),
            (
                "unmeasured or shared",
                image(vec![
                    flagged(IgvmPageDataFlags::new().with_unmeasured(true)),
                    vmsa(),
                ]),
            ),
            (
                "unmeasured or shared",
                image(vec![
                    flagged(IgvmPageDataFlags::new().with_shared(true)),
                    vmsa(),
                ]),
            ),
            (
                "unsupported data type",
                image(vec![
                    page(0x1000, 1),
                    typed(0x2000, IgvmPageDataType::CPUID_XF),
                    vmsa(),
                ]),
            ),
            (
                "not 2 MiB-aligned",
                image(vec![
                    page(0x1000, 1),
                    IgvmDirectiveHeader::PageData {
                        gpa: 0x20_1000,
                        compatibility_mask: SNP_MASK,
                        flags: IgvmPageDataFlags::new().with_is_2mb_page(true),
                        data_type: IgvmPageDataType::NORMAL,
                        data: vec![],
                    },
                    vmsa(),
                ]),
            ),
            (
                "inserted after page data",
                image(vec![
                    IgvmDirectiveHeader::ParameterArea {
                        number_of_bytes: PAGE_SIZE,
                        parameter_area_index: 0,
                        initial_data: vec![],
                    },
                    page(0x2000, 1),
                    IgvmDirectiveHeader::ParameterInsert(IGVM_VHS_PARAMETER_INSERT {
                        gpa: 0x5000,
                        compatibility_mask: SNP_MASK,
                        parameter_area_index: 0,
                    }),
                    vmsa(),
                ]),
            ),
        ];
        for (needle, bytes) in cases {
            let err = launch_digest_from_igvm(&bytes).unwrap_err().to_string();
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }

    fn manifest(igvm: &[u8], measurement: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "version": "v0.0.0-test",
            "root": "00",
            "igvm": {
                "format_version": 1,
                "snp": hex::encode(Sha256::digest(igvm)),
                "tdx": "00",
                "snp_launch": {"measurement": measurement},
                "tdx_launch": {},
            },
        }))
        .unwrap()
    }

    fn config(manifest: &[u8]) -> Vec<u8> {
        format!(
            "cvm-version: 0.0.0-test@sha256:{}\ncpus: 2\n",
            hex::encode(Sha256::digest(manifest))
        )
        .into_bytes()
    }

    #[test]
    fn release_pin_requires_every_input_to_agree() {
        let igvm = image(vec![page(0x2000, 1), vmsa()]);
        let measurement = hex::encode(launch_digest_from_igvm(&igvm).unwrap().measurement);
        let good = manifest(&igvm, &measurement);
        let cfg = config(&good);

        let pin = release_pin(&good, &igvm, &cfg).unwrap();
        assert_eq!(pin.sev_snp.measurement, measurement);
        assert_eq!(pin.sev_snp.host_data, hex::encode(Sha256::digest(&cfg)));
        assert_eq!(pin.sev_snp.guest_policy, "0x30133");
        assert_eq!(pin.igvm_sha256, hex::encode(Sha256::digest(&igvm)));
        assert_eq!(
            pin.cvm_version,
            format!("0.0.0-test@sha256:{}", hex::encode(Sha256::digest(&good)))
        );

        let err = |manifest: &[u8], igvm: &[u8], config: &[u8]| {
            release_pin(manifest, igvm, config).unwrap_err().to_string()
        };
        // No inline pin, or a pin naming another manifest.
        assert!(err(&good, &igvm, b"cvm-version: 0.0.0-test\n").contains("pin its release"));
        assert!(err(&good, &igvm, &config(b"other")).contains("config pins release manifest"));
        // Another image than the manifest records.
        let other = image(vec![page(0x2000, 2), vmsa()]);
        assert!(err(&good, &other, &cfg).contains("records"));
        // A manifest stating another digest.
        let wrong = manifest(&igvm, &"11".repeat(48));
        assert!(err(&wrong, &igvm, &config(&wrong)).contains("disagrees"));
        // A config naming another release than the manifest describes.
        let mut renamed: serde_json::Value = serde_json::from_slice(&good).unwrap();
        renamed["version"] = "v0.0.1-test".into();
        let renamed = serde_json::to_vec(&renamed).unwrap();
        assert!(err(&renamed, &igvm, &config(&renamed)).contains("config selects cvm-version"));
        // A manifest layout this tool does not know.
        let mut future: serde_json::Value = serde_json::from_slice(&good).unwrap();
        future["igvm"]["format_version"] = 2.into();
        let future = serde_json::to_vec(&future).unwrap();
        assert!(err(&future, &igvm, &config(&future)).contains("format_version"));
    }

    #[test]
    fn host_data_is_the_config_sha256_and_the_policy_has_one_spelling() {
        assert_eq!(
            hex::encode(host_data(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(guest_policy_hex(0x30133), "0x30133");
        assert_eq!(guest_policy_hex(0x1), "0x1");
    }

    /// Published SEV-SNP images and the launch digest each release manifest
    /// states: `(file name, image SHA-256, digest)`. The images (about 9 MB
    /// of kernel and initramfs each) are not committed; the hashes come from
    /// each release's `tinfoil-inference-<version>-manifest.json`
    /// (`igvm.snp`, `igvm.snp_launch.measurement`).
    const PUBLISHED: &[(&str, &str, &str)] = &[
        (
            "tinfoil-snp-v0.15.3-rc.1.igvm",
            "8fe54c46ceb111b7cc41f12f60bd479211870c7bc5186c8566d71f286ad0d5ac",
            "d712705df47368b82a283ba74501c5cf807f5a099c67906a5e1d861037514b9e\
             fe382024d4eb6245bb8c9877c7fe6858",
        ),
        (
            "tinfoil-snp-v0.15.3-rc.1+gpu.igvm",
            "eacdbedae52b674e1fb71e3860aec72af85f4a55c51304f718ec4f47ce9c683a",
            "605c0b29196443fe02c3fc9b6c0bf4efda8948dc93d32a1bbcb25da1f969eda6\
             ddbdd08d9e5358c9a9bf57f2b0c0ab98",
        ),
    ];

    /// Known answers against published releases: set `SNP_IGVM_DIR` to a
    /// directory holding the images [`PUBLISHED`] names (from
    /// `https://images.tinfoil.sh/cvm/<file name>`). Each image present must
    /// hash to its manifest's record, reproduce its digest and ask for guest
    /// policy `0x30133`; at least one must be present.
    #[test]
    fn published_release_launch_digest_known_answer() {
        let Ok(dir) = std::env::var("SNP_IGVM_DIR") else {
            eprintln!("skipping: set SNP_IGVM_DIR to run");
            return;
        };
        let mut checked = 0;
        for (name, sha256, measurement) in PUBLISHED {
            let Ok(bytes) = std::fs::read(std::path::Path::new(&dir).join(name)) else {
                eprintln!("{name}: not in {dir}");
                continue;
            };
            assert_eq!(hex::encode(Sha256::digest(&bytes)), *sha256, "{name}");
            let image = launch_digest_from_igvm(&bytes).unwrap();
            assert_eq!(hex::encode(image.measurement), *measurement, "{name}");
            assert_eq!(image.guest_policy, POLICY, "{name}");
            checked += 1;
        }
        assert!(checked > 0, "no published image in {dir}");
    }
}
