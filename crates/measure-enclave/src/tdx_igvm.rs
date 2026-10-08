//! The Intel TDX launch identity of a Tinfoil CVM in the IGVM launch model
//! (cvmimage v0.15 and later), computed from public inputs.
//!
//! In that model the guest is a single IGVM image with no firmware and no
//! measured boot after launch. Its TDX identity is two registers:
//!
//! - **MRTD**, which the TDX module accumulates over every page the host adds
//!   to the trust domain before it runs. It is a function of the IGVM image
//!   alone and is the same for every machine shape, so each cvmimage release
//!   has exactly one. [`mrtd_from_igvm`] recomputes it from the image bytes.
//! - **MRCONFIGID**, which the host passes at launch and the guest checks
//!   against its own config before booting: the SHA-256 of the deployment's
//!   `tinfoil-config.yml` followed by sixteen zero bytes ([`mrconfigid`]).
//!
//! RTMR0–RTMR3 are zero at launch and the image extends none of them.
//!
//! [`release_pin`] ties the computation to a release: the release manifest is
//! pinned by SHA-256, the IGVM image must hash to what that manifest records,
//! and the recomputed MRTD must equal the MRTD the manifest states. A
//! disagreement anywhere is an error rather than a choice between values.

use anyhow::{Context, Result, bail, ensure};
use igvm::{IgvmDirectiveHeader, IgvmFile, IgvmPlatformHeader};
use igvm_defs::{IgvmPageDataType, IgvmPlatformType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, Sha384};

const PAGE_SIZE: u64 = 4096;
/// TDH.MR.EXTEND measures a page in 256-byte chunks.
const MR_EXTEND_CHUNK: usize = 256;
/// Each MRTD update starts with a 128-byte record: an ASCII operation name,
/// zero-padded, with the guest-physical address as a little-endian `u64` at
/// byte 16.
const MR_RECORD_LEN: usize = 128;
const MR_RECORD_GPA: usize = 16;

/// MRTD accumulator following the TDX module's definition: one
/// `MEM.PAGE.ADD` record per page added, then, for a measured page, one
/// `MR.EXTEND` record plus 256 bytes of content per chunk.
struct Mrtd(Sha384);

impl Mrtd {
    fn record(&mut self, operation: &[u8], gpa: u64) {
        let mut record = [0u8; MR_RECORD_LEN];
        record[..operation.len()].copy_from_slice(operation);
        record[MR_RECORD_GPA..MR_RECORD_GPA + 8].copy_from_slice(&gpa.to_le_bytes());
        self.0.update(record);
    }

    fn add_page(&mut self, gpa: u64) {
        self.record(b"MEM.PAGE.ADD", gpa);
    }

    fn extend_page(&mut self, gpa: u64, content: &[u8; PAGE_SIZE as usize]) {
        for (index, chunk) in content.as_chunks::<MR_EXTEND_CHUNK>().0.iter().enumerate() {
            self.record(b"MR.EXTEND", gpa + (index * MR_EXTEND_CHUNK) as u64);
            self.0.update(chunk);
        }
    }
}

/// Recompute MRTD for a TDX IGVM image.
///
/// Directives are applied in file order, as a loader imports them. Measured
/// page data is added and extended; unmeasured page data and inserted
/// parameter areas are added only, so their addresses reach MRTD and their
/// contents do not; shared pages are never added to the private address
/// space. Directives that only describe parameters or memory the host
/// supplies after measurement contribute nothing. Anything this
/// computation does not model — large pages, non-normal page types, an
/// explicit initial VP context — is refused rather than guessed at.
pub fn mrtd_from_igvm(bytes: &[u8]) -> Result<[u8; 48]> {
    let file = IgvmFile::new_from_binary(bytes, None).context("parsing IGVM image")?;
    let tdx_mask = match file.platforms() {
        [IgvmPlatformHeader::SupportedPlatform(platform)]
            if platform.platform_type == IgvmPlatformType::TDX =>
        {
            platform.compatibility_mask
        }
        other => bail!(
            "IGVM image must declare exactly one platform, TDX; it declares {}",
            other.len()
        ),
    };

    let mut parameter_areas = std::collections::HashMap::new();
    let mut mrtd = Mrtd(Sha384::new());
    for directive in file.directives() {
        match directive {
            IgvmDirectiveHeader::PageData {
                gpa,
                compatibility_mask,
                flags,
                data_type,
                data,
            } => {
                if compatibility_mask & tdx_mask == 0 {
                    continue;
                }
                ensure!(
                    !flags.is_2mb_page(),
                    "2 MiB page at {gpa:#x} is not supported"
                );
                ensure!(
                    *data_type == IgvmPageDataType::NORMAL,
                    "page at {gpa:#x} has a non-normal data type"
                );
                if flags.shared() {
                    continue;
                }
                mrtd.add_page(*gpa);
                if !flags.unmeasured() {
                    let mut content = [0u8; PAGE_SIZE as usize];
                    ensure!(
                        data.is_empty() || data.len() == content.len(),
                        "page at {gpa:#x} carries {} bytes",
                        data.len()
                    );
                    content[..data.len()].copy_from_slice(data);
                    mrtd.extend_page(*gpa, &content);
                }
            }
            IgvmDirectiveHeader::ParameterArea {
                number_of_bytes,
                parameter_area_index,
                ..
            } => {
                parameter_areas.insert(*parameter_area_index, *number_of_bytes);
            }
            IgvmDirectiveHeader::ParameterInsert(insert) => {
                if insert.compatibility_mask & tdx_mask == 0 {
                    continue;
                }
                let size = parameter_areas
                    .get(&insert.parameter_area_index)
                    .with_context(|| {
                        format!(
                            "parameter area {} inserted before it is declared",
                            insert.parameter_area_index
                        )
                    })?;
                for page in 0..size.div_ceil(PAGE_SIZE) {
                    mrtd.add_page(insert.gpa + page * PAGE_SIZE);
                }
            }
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
            other => bail!("IGVM directive is outside the TDX measurement model: {other:?}"),
        }
    }
    Ok(mrtd.0.finalize().into())
}

/// MRCONFIGID for a deployment config: SHA-256 of the exact config bytes,
/// followed by sixteen zero bytes.
pub fn mrconfigid(config: &[u8]) -> [u8; 48] {
    let mut field = [0u8; 48];
    field[..32].copy_from_slice(&Sha256::digest(config));
    field
}

/// The fields of a cvmimage release manifest this tool reads.
#[derive(Deserialize)]
struct ReleaseManifest {
    version: String,
    igvm: ReleaseIgvm,
}

#[derive(Deserialize)]
struct ReleaseIgvm {
    /// SHA-256 of the TDX IGVM image.
    tdx: String,
    tdx_launch: TdxLaunch,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TdxLaunch {
    mrtd: String,
    rtmr0: String,
    rtmr1: String,
    rtmr2: String,
    rtmr3: String,
}

/// The TDX launch identity of one deployment of one release.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct TdxLaunchPin {
    /// The cvmimage release the image came from.
    pub cvm_version: String,
    /// SHA-256 of the IGVM image MRTD was computed from.
    pub igvm_sha256: String,
    /// Hex-encoded MRTD.
    pub mrtd: String,
    /// Hex-encoded MRCONFIGID.
    pub mrconfigid: String,
}

/// Compute a deployment's TDX launch identity and check it against the
/// release that published the image.
pub fn release_pin(
    manifest: &[u8],
    manifest_sha256: &str,
    igvm: &[u8],
    config: &[u8],
) -> Result<TdxLaunchPin> {
    let actual = hex::encode(Sha256::digest(manifest));
    ensure!(
        actual == manifest_sha256.to_ascii_lowercase(),
        "release manifest SHA-256 is {actual}, not the pinned {manifest_sha256}"
    );
    let manifest: ReleaseManifest =
        serde_json::from_slice(manifest).context("parsing release manifest")?;
    let launch = &manifest.igvm.tdx_launch;
    let zero = "00".repeat(48);
    for (name, value) in [
        ("rtmr0", &launch.rtmr0),
        ("rtmr1", &launch.rtmr1),
        ("rtmr2", &launch.rtmr2),
        ("rtmr3", &launch.rtmr3),
    ] {
        ensure!(
            *value == zero,
            "release states a non-zero TDX {name}; it is not an IGVM-model release"
        );
    }

    let igvm_sha256 = hex::encode(Sha256::digest(igvm));
    ensure!(
        igvm_sha256 == manifest.igvm.tdx.to_ascii_lowercase(),
        "IGVM image SHA-256 is {igvm_sha256}, but release {} records {}",
        manifest.version,
        manifest.igvm.tdx
    );
    let mrtd = hex::encode(mrtd_from_igvm(igvm)?);
    ensure!(
        mrtd == launch.mrtd.to_ascii_lowercase(),
        "recomputed MRTD {mrtd} disagrees with the MRTD release {} states, {}",
        manifest.version,
        launch.mrtd
    );
    Ok(TdxLaunchPin {
        cvm_version: manifest.version,
        igvm_sha256,
        mrtd,
        mrconfigid: hex::encode(mrconfigid(config)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use igvm::IgvmRevision;
    use igvm_defs::{IGVM_VHS_PARAMETER, IGVM_VHS_PARAMETER_INSERT, IGVM_VHS_SUPPORTED_PLATFORM};
    use igvm_defs::{IgvmPageDataFlags, IgvmPlatformType};

    const TDX_MASK: u32 = 0x1;

    fn image(directives: Vec<IgvmDirectiveHeader>) -> Vec<u8> {
        let file = IgvmFile::new(
            IgvmRevision::V1,
            vec![IgvmPlatformHeader::SupportedPlatform(
                IGVM_VHS_SUPPORTED_PLATFORM {
                    compatibility_mask: TDX_MASK,
                    highest_vtl: 0,
                    platform_type: IgvmPlatformType::TDX,
                    platform_version: 1,
                    shared_gpa_boundary: 0,
                },
            )],
            vec![],
            directives,
        )
        .unwrap();
        let mut out = Vec::new();
        file.serialize(&mut out).unwrap();
        out
    }

    fn page(gpa: u64, fill: u8, flags: IgvmPageDataFlags) -> IgvmDirectiveHeader {
        IgvmDirectiveHeader::PageData {
            gpa,
            compatibility_mask: TDX_MASK,
            flags,
            data_type: IgvmPageDataType::NORMAL,
            data: vec![fill; PAGE_SIZE as usize],
        }
    }

    /// The accumulation written out longhand from the TDX module's record
    /// layout, as an independent check of [`Mrtd`].
    fn longhand(ops: &[(&str, u64, Option<u8>)]) -> [u8; 48] {
        let mut hash = Sha384::new();
        for (op, gpa, fill) in ops {
            let mut add = [0u8; 128];
            add[..op.len()].copy_from_slice(op.as_bytes());
            add[16..24].copy_from_slice(&gpa.to_le_bytes());
            hash.update(add);
            if let Some(fill) = fill {
                for chunk in 0..16u64 {
                    let mut ext = [0u8; 128];
                    ext[..9].copy_from_slice(b"MR.EXTEND");
                    ext[16..24].copy_from_slice(&(gpa + chunk * 256).to_le_bytes());
                    hash.update(ext);
                    hash.update([*fill; 256]);
                }
            }
        }
        hash.finalize().into()
    }

    #[test]
    fn mrtd_adds_every_private_page_and_extends_only_measured_ones() {
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
                gpa: 0x1000,
                compatibility_mask: TDX_MASK,
                parameter_area_index: 0,
            }),
            page(0x2000, 0xaa, IgvmPageDataFlags::new()),
            page(0x3000, 0xbb, IgvmPageDataFlags::new().with_unmeasured(true)),
            page(0x4000, 0xcc, IgvmPageDataFlags::new().with_shared(true)),
            page(0x5000, 0xdd, IgvmPageDataFlags::new()),
        ]);
        let expected = longhand(&[
            ("MEM.PAGE.ADD", 0x1000, None),
            ("MEM.PAGE.ADD", 0x2000, Some(0xaa)),
            ("MEM.PAGE.ADD", 0x3000, None),
            ("MEM.PAGE.ADD", 0x5000, Some(0xdd)),
        ]);
        assert_eq!(mrtd_from_igvm(&bytes).unwrap(), expected);
    }

    #[test]
    fn mrtd_depends_on_order_address_and_content() {
        let base = mrtd_from_igvm(&image(vec![
            page(0x2000, 1, IgvmPageDataFlags::new()),
            page(0x3000, 2, IgvmPageDataFlags::new()),
        ]))
        .unwrap();
        for variant in [
            vec![
                page(0x3000, 2, IgvmPageDataFlags::new()),
                page(0x2000, 1, IgvmPageDataFlags::new()),
            ],
            vec![
                page(0x2000, 1, IgvmPageDataFlags::new()),
                page(0x4000, 2, IgvmPageDataFlags::new()),
            ],
            vec![
                page(0x2000, 1, IgvmPageDataFlags::new()),
                page(0x3000, 3, IgvmPageDataFlags::new()),
            ],
        ] {
            assert_ne!(mrtd_from_igvm(&image(variant)).unwrap(), base);
        }
    }

    #[test]
    fn mrconfigid_pads_the_config_hash() {
        assert_eq!(
            hex::encode(mrconfigid(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\
             00000000000000000000000000000000"
        );
    }

    fn manifest(igvm: &[u8], mrtd: &str, rtmr0: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "version": "v0.0.0-test",
            "root": "00",
            "igvm": {
                "format_version": 1,
                "tdx": hex::encode(Sha256::digest(igvm)),
                "tdx_launch": {
                    "mrtd": mrtd,
                    "rtmr0": rtmr0,
                    "rtmr1": "00".repeat(48),
                    "rtmr2": "00".repeat(48),
                    "rtmr3": "00".repeat(48),
                },
            },
        }))
        .unwrap()
    }

    #[test]
    fn release_pin_requires_every_input_to_agree() {
        let igvm = image(vec![page(0x2000, 1, IgvmPageDataFlags::new())]);
        let mrtd = hex::encode(mrtd_from_igvm(&igvm).unwrap());
        let zero = "00".repeat(48);
        let good = manifest(&igvm, &mrtd, &zero);
        let good_sha = hex::encode(Sha256::digest(&good));

        let pin = release_pin(&good, &good_sha, &igvm, b"config").unwrap();
        assert_eq!(pin.mrtd, mrtd);
        assert_eq!(pin.mrconfigid, hex::encode(mrconfigid(b"config")));
        assert_eq!(pin.cvm_version, "v0.0.0-test");

        let err = |manifest: &[u8], sha: &str, igvm: &[u8]| {
            release_pin(manifest, sha, igvm, b"config")
                .unwrap_err()
                .to_string()
        };
        assert!(err(&good, &"00".repeat(32), &igvm).contains("not the pinned"));
        let other_image = image(vec![page(0x2000, 2, IgvmPageDataFlags::new())]);
        assert!(err(&good, &good_sha, &other_image).contains("records"));
        let wrong_mrtd = manifest(&igvm, &"11".repeat(48), &zero);
        let sha = hex::encode(Sha256::digest(&wrong_mrtd));
        assert!(err(&wrong_mrtd, &sha, &igvm).contains("disagrees"));
        let measured_boot = manifest(&igvm, &mrtd, &"22".repeat(48));
        let sha = hex::encode(Sha256::digest(&measured_boot));
        assert!(err(&measured_boot, &sha, &igvm).contains("rtmr0"));
    }

    /// Known answer against a published release: set `TDX_IGVM` to the
    /// release's `tinfoil-tdx-<version>.igvm` and `TDX_IGVM_MRTD` to the
    /// MRTD its manifest states.
    #[test]
    fn published_release_mrtd_known_answer() {
        let (Ok(path), Ok(expected)) = (std::env::var("TDX_IGVM"), std::env::var("TDX_IGVM_MRTD"))
        else {
            eprintln!("skipping: set TDX_IGVM and TDX_IGVM_MRTD to run");
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(hex::encode(mrtd_from_igvm(&bytes).unwrap()), expected);
    }
}
