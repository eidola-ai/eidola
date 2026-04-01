//! Compute deterministic SEV-SNP and TDX measurements for a Tinfoil Container.
//!
//! Given a `tinfoil-config.yml` and the CVM image artifacts (OVMF firmware,
//! kernel, initrd), this tool pre-computes the hardware attestation measurements
//! that a legitimate enclave will produce. These measurements are committed in
//! `artifact-manifest.json` so clients can verify enclaves without trusting any
//! third party.
//!
//! The measurement is a deterministic function of:
//!   1. OVMF firmware binary
//!   2. Linux kernel (vmlinuz)
//!   3. initrd
//!   4. Kernel command line (embeds dm-verity roothash + SHA-256 of config)
//!   5. vCPU count and type
//!
//! Usage:
//!   cargo run -p measure-enclave -- \
//!     --config tinfoil-config.yml \
//!     --ovmf /tmp/cvm/OVMF.fd \
//!     --kernel /tmp/cvm/vmlinuz \
//!     --initrd /tmp/cvm/initrd \
//!     --roothash abc123...

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use sev::measurement::{
    snp::{SnpMeasurementArgs, snp_calc_launch_digest},
    vcpu_types::CpuType,
    vmsa::{GuestFeatures, VMMType},
};

use tdx_measure::{Machine, TdxMeasurements};

#[derive(Parser)]
#[command(about = "Compute enclave measurements for Tinfoil Containers")]
struct Cli {
    /// Path to tinfoil-config.yml
    #[arg(long)]
    config: PathBuf,

    /// Path to OVMF firmware binary
    #[arg(long)]
    ovmf: PathBuf,

    /// Path to kernel (vmlinuz)
    #[arg(long)]
    kernel: PathBuf,

    /// Path to initrd
    #[arg(long)]
    initrd: PathBuf,

    /// dm-verity roothash from CVM image manifest (hex)
    #[arg(long)]
    roothash: String,
}

/// Minimal parse of tinfoil-config.yml — only the fields we need.
#[derive(Deserialize)]
#[allow(dead_code)]
struct TinfoilConfig {
    cpus: u32,
    memory: u64,
}

/// Output format — matches what we store in artifact-manifest.json.
#[derive(Serialize)]
struct MeasurementOutput {
    snp_measurement: String,
    tdx_rtmr1: String,
    tdx_rtmr2: String,
}

/// Build the kernel command line exactly as the Tinfoil CVM does.
///
/// This must match `measure.py` in `tinfoilsh/measure-image-action`:
///   readonly=on pci=realloc,nocrs modprobe.blacklist=nouveau nouveau.modeset=0
///   root=/dev/mapper/root roothash=<ROOTHASH> tinfoil-config-hash=<CONFIG_HASH>
fn build_cmdline(roothash: &str, config_path: &Path) -> Result<String> {
    let config_bytes =
        std::fs::read(config_path).with_context(|| format!("reading {}", config_path.display()))?;
    let config_hash = hex::encode(Sha256::digest(&config_bytes));

    Ok(format!(
        "readonly=on pci=realloc,nocrs modprobe.blacklist=nouveau nouveau.modeset=0 \
         root=/dev/mapper/root roothash={roothash} tinfoil-config-hash={config_hash}"
    ))
}

fn compute_snp(
    ovmf: &Path,
    kernel: &Path,
    initrd: &Path,
    cmdline: &str,
    vcpus: u32,
) -> Result<String> {
    let args = SnpMeasurementArgs {
        vcpus,
        vcpu_type: CpuType::EpycV4,
        ovmf_file: ovmf.to_path_buf(),
        guest_features: GuestFeatures(0x1), // SNPActive
        kernel_file: Some(kernel.to_path_buf()),
        initrd_file: Some(initrd.to_path_buf()),
        append: Some(cmdline),
        ovmf_hash_str: None,
        vmm_type: Some(VMMType::QEMU),
    };

    let digest = snp_calc_launch_digest(args).context("SNP measurement failed")?;
    Ok(digest.get_hex_ld())
}

fn compute_tdx(kernel: &Path, initrd: &Path, cmdline: &str) -> Result<TdxMeasurements> {
    let kernel_str = kernel.to_str().context("kernel path not valid UTF-8")?;
    let initrd_str = initrd.to_str().context("initrd path not valid UTF-8")?;

    // measure_runtime() only accesses kernel, initrd, kernel_cmdline, and
    // direct_boot. Platform fields (firmware, acpi_tables, rsdp, etc.) are
    // unused for runtime-only measurement, so we pass empty values.
    let machine = Machine {
        cpu_count: 0,   // unused by measure_runtime
        memory_size: 0, // unused — hardcoded to 0xb0000000
        qcow2: None,
        firmware: "",
        kernel: Some(kernel_str),
        initrd: Some(initrd_str),
        kernel_cmdline: cmdline,
        acpi_tables: "",
        rsdp: "",
        table_loader: "",
        boot_order: "",
        path_boot_xxxx: "",
        mok_list: None,
        mok_list_trusted: None,
        mok_list_x: None,
        sbat_level: None,
        direct_boot: true,
    };

    machine.measure_runtime().context("TDX measurement failed")
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let config: TinfoilConfig = serde_yaml::from_reader(
        std::fs::File::open(&cli.config)
            .with_context(|| format!("opening {}", cli.config.display()))?,
    )
    .context("parsing tinfoil-config.yml")?;

    let cmdline = build_cmdline(&cli.roothash, &cli.config)?;

    let snp = compute_snp(&cli.ovmf, &cli.kernel, &cli.initrd, &cmdline, config.cpus)?;
    let tdx = compute_tdx(&cli.kernel, &cli.initrd, &cmdline)?;

    let output = MeasurementOutput {
        snp_measurement: snp,
        tdx_rtmr1: hex::encode(&tdx.rtmr1),
        tdx_rtmr2: hex::encode(&tdx.rtmr2),
    };

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}
