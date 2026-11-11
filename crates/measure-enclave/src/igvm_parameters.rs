//! The parameter areas an IGVM image asks the host to fill at launch.
//!
//! A parameter area is guest memory the loader writes when it launches the
//! image, and its insertion is an unmeasured page: the launch measurement
//! (MRTD, or the SEV-SNP launch digest) covers its address and never its
//! contents. Whatever the guest reads there is the host's choice under an
//! unchanged measurement. An area is accepted only when every parameter the
//! image places in it describes the machine shape, which the host chooses at
//! launch anyway and the launch identity excludes by design: the processor
//! count and the memory map, the only parameters a measured release declares.
//! Every other parameter is refused: a kernel command line or a device tree
//! is boot configuration, and the host-built ACPI tables, MMIO ranges and
//! environment information are guest inputs no measured release has asked
//! for, to be reviewed before one is admitted. An area carrying initial data
//! (image content the host may replace) is refused, and so is an inserted
//! area no parameter is placed in, since nothing then says what the guest
//! reads from it.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use igvm::IgvmDirectiveHeader;

const PAGE_SIZE: u64 = 4096;

/// A parameter area insertion to measure: `pages` unmeasured 4 KiB pages
/// from `gpa`.
pub(crate) struct Insert {
    pub(crate) index: u32,
    pub(crate) gpa: u64,
    pub(crate) pages: u64,
}

/// What a directive does to the parameter areas.
pub(crate) enum Step {
    /// The directive is not about parameter areas.
    Other,
    /// Accounted for; nothing to measure on this platform.
    Recorded,
    /// An area is inserted on this platform.
    Insert(Insert),
}

struct Area {
    bytes: u64,
    holds_parameter: bool,
}

/// The parameter areas declared so far, in file order.
#[derive(Default)]
pub(crate) struct ParameterAreas(HashMap<u32, Area>);

impl ParameterAreas {
    /// Account for one directive; `platform_mask` is the measured platform's
    /// compatibility mask.
    pub(crate) fn apply(
        &mut self,
        directive: &IgvmDirectiveHeader,
        platform_mask: u32,
    ) -> Result<Step> {
        let (what, parameter) = match directive {
            IgvmDirectiveHeader::ParameterArea {
                number_of_bytes,
                parameter_area_index,
                initial_data,
            } => {
                ensure!(
                    initial_data.is_empty(),
                    "parameter area {parameter_area_index} carries initial data"
                );
                self.0.insert(
                    *parameter_area_index,
                    Area {
                        bytes: *number_of_bytes,
                        holds_parameter: false,
                    },
                );
                return Ok(Step::Recorded);
            }
            IgvmDirectiveHeader::VpCount(parameter) | IgvmDirectiveHeader::MemoryMap(parameter) => {
                self.0
                    .get_mut(&parameter.parameter_area_index)
                    .with_context(|| {
                        format!(
                            "parameter area {} used before it is declared",
                            parameter.parameter_area_index
                        )
                    })?
                    .holds_parameter = true;
                return Ok(Step::Recorded);
            }
            IgvmDirectiveHeader::ParameterInsert(insert) => {
                if insert.compatibility_mask & platform_mask == 0 {
                    return Ok(Step::Recorded);
                }
                let index = insert.parameter_area_index;
                let area = self.0.get(&index).with_context(|| {
                    format!("parameter area {index} inserted before it is declared")
                })?;
                ensure!(
                    area.holds_parameter,
                    "parameter area {index} is inserted at {:#x} with no parameter placed in \
                     it; its contents would be the host's, for an unknown purpose",
                    insert.gpa
                );
                return Ok(Step::Insert(Insert {
                    index,
                    gpa: insert.gpa,
                    pages: area.bytes.div_ceil(PAGE_SIZE),
                }));
            }
            IgvmDirectiveHeader::CommandLine(p) => ("a kernel command line", p),
            IgvmDirectiveHeader::DeviceTree(p) => ("a device tree", p),
            IgvmDirectiveHeader::EnvironmentInfo(p) => ("environment information", p),
            IgvmDirectiveHeader::Srat(p) => ("an ACPI SRAT", p),
            IgvmDirectiveHeader::Madt(p) => ("an ACPI MADT", p),
            IgvmDirectiveHeader::Slit(p) => ("an ACPI SLIT", p),
            IgvmDirectiveHeader::Pptt(p) => ("an ACPI PPTT", p),
            IgvmDirectiveHeader::MmioRanges(p) => ("MMIO ranges", p),
            _ => return Ok(Step::Other),
        };
        bail!(
            "IGVM image asks the host for {what} (parameter area {}); the host would choose it \
             under an unchanged launch measurement",
            parameter.parameter_area_index
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tdx_igvm::{
        mrtd_from_igvm,
        tests::{TDX_MASK, image, page},
    };
    use igvm::IgvmDirectiveHeader;
    use igvm_defs::{IGVM_VHS_PARAMETER, IGVM_VHS_PARAMETER_INSERT, IgvmPageDataFlags};

    fn area(initial_data: Vec<u8>) -> IgvmDirectiveHeader {
        IgvmDirectiveHeader::ParameterArea {
            number_of_bytes: 4096,
            parameter_area_index: 0,
            initial_data,
        }
    }

    fn insert() -> IgvmDirectiveHeader {
        IgvmDirectiveHeader::ParameterInsert(IGVM_VHS_PARAMETER_INSERT {
            gpa: 0x1000,
            compatibility_mask: TDX_MASK,
            parameter_area_index: 0,
        })
    }

    const AREA_0: IGVM_VHS_PARAMETER = IGVM_VHS_PARAMETER {
        parameter_area_index: 0,
        byte_offset: 0,
    };

    #[test]
    fn only_the_machine_shape_may_be_left_to_the_host() {
        let with = |parameter: IgvmDirectiveHeader| {
            mrtd_from_igvm(&image(vec![
                area(vec![]),
                parameter,
                insert(),
                page(0x2000, 1, IgvmPageDataFlags::new()),
            ]))
        };
        with(IgvmDirectiveHeader::VpCount(AREA_0)).expect("processor count");
        with(IgvmDirectiveHeader::MemoryMap(AREA_0)).expect("memory map");
        for (parameter, needle) in [
            (
                IgvmDirectiveHeader::CommandLine(AREA_0),
                "kernel command line",
            ),
            (IgvmDirectiveHeader::DeviceTree(AREA_0), "device tree"),
            (
                IgvmDirectiveHeader::EnvironmentInfo(AREA_0),
                "environment information",
            ),
            (IgvmDirectiveHeader::Srat(AREA_0), "SRAT"),
            (IgvmDirectiveHeader::Madt(AREA_0), "MADT"),
            (IgvmDirectiveHeader::Slit(AREA_0), "SLIT"),
            (IgvmDirectiveHeader::Pptt(AREA_0), "PPTT"),
            (IgvmDirectiveHeader::MmioRanges(AREA_0), "MMIO ranges"),
        ] {
            let err = with(parameter).unwrap_err().to_string();
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }

    #[test]
    fn an_area_with_initial_data_or_no_parameter_is_refused() {
        let page = || page(0x2000, 1, IgvmPageDataFlags::new());
        let err = mrtd_from_igvm(&image(vec![
            area(vec![1; 4096]),
            IgvmDirectiveHeader::VpCount(AREA_0),
            insert(),
            page(),
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("carries initial data"), "{err}");
        let err = mrtd_from_igvm(&image(vec![area(vec![]), insert(), page()]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no parameter placed in it"), "{err}");
    }
}
