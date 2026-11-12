//! Compute the SEV-SNP launch identity of the Eidola gateway's Tinfoil
//! Container from public inputs only.
//!
//! The gateway runs on the platform provider's IGVM launch model: one
//! firmware-less SEV-SNP image per release, whose launch digest is a function
//! of the image alone, and the deployment's config bound into every report
//! through HOST_DATA. The output is the `releases/trust/server-enclave.json`
//! record less its `schema_version` envelope (`scripts/artifact-manifest.sh`
//! adds it), and the `enclave` block of `artifact-manifest.json`.
//!
//! Usage:
//!   cargo run -p measure-enclave -- \
//!     --config tinfoil-config.yml \
//!     --release-manifest tinfoil-inference-vX-manifest.json \
//!     --igvm tinfoil-snp-vX.igvm
//!
//! The config must pin the release manifest inline (`cvm-version:
//! X@sha256:<hex>`) and select the release's CPU image (no `+` build tag: the
//! gateway attaches no GPUs). The manifest must hash to that pin, the image to
//! the manifest's `igvm.snp`, and the recomputed launch digest must equal the
//! manifest's `igvm.snp_launch.measurement`; see
//! `measure_enclave::snp_igvm::release_pin`.

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Compute the SEV-SNP launch identity of the Eidola gateway enclave")]
struct Cli {
    /// The gateway's `tinfoil-config.yml`, exactly as deployed.
    #[arg(long)]
    config: PathBuf,
    /// cvmimage release manifest (`tinfoil-inference-v<version>-manifest.json`).
    #[arg(long)]
    release_manifest: PathBuf,
    /// The release's SEV-SNP IGVM image (`tinfoil-snp-v<version>.igvm`).
    #[arg(long)]
    igvm: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let read =
        |path: &PathBuf| std::fs::read(path).with_context(|| format!("reading {}", path.display()));
    let config = read(&cli.config)?;
    let release = measure_enclave::tdx_igvm::config_release(&config)?;
    ensure!(
        !release.version.contains('+'),
        "cvm-version {} names a tagged image; the gateway runs on the release's plain CPU image",
        release.version
    );
    let pin = measure_enclave::snp_igvm::release_pin(
        &read(&cli.release_manifest)?,
        &read(&cli.igvm)?,
        &config,
    )?;
    println!("{}", serde_json::to_string_pretty(&pin)?);
    Ok(())
}
