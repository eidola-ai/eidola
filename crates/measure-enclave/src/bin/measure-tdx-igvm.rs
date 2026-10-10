//! Compute the Intel TDX launch identity (MRTD + MRCONFIGID) of a Tinfoil
//! CVM in the IGVM launch model, from public inputs only.
//!
//! Usage:
//!   cargo run -p measure-enclave --bin measure-tdx-igvm -- \
//!     --release-manifest tinfoil-inference-vX-manifest.json \
//!     --release-manifest-sha256 <hex> \
//!     --igvm tinfoil-tdx-vX.igvm \
//!     --config tinfoil-config.yml
//!
//! The release manifest is pinned by SHA-256; the IGVM image must hash to
//! the value that manifest records; MRTD is recomputed from the image and
//! must equal the MRTD the manifest states. MRCONFIGID is computed from the
//! config bytes. The output is the launch half of a verifier TDX pin; the
//! machine policy (MR_SEAM, attributes, TCB floors) is not a function of the
//! image and is not emitted here.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Compute the TDX launch identity of an IGVM-model Tinfoil CVM")]
struct Cli {
    /// cvmimage release manifest (`tinfoil-inference-<version>-manifest.json`).
    #[arg(long)]
    release_manifest: PathBuf,
    /// Pinned SHA-256 of the release manifest (hex).
    #[arg(long)]
    release_manifest_sha256: String,
    /// The release's TDX IGVM image (`tinfoil-tdx-<version>.igvm`).
    #[arg(long)]
    igvm: PathBuf,
    /// The deployment's `tinfoil-config.yml`, exactly as deployed.
    #[arg(long)]
    config: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let read =
        |path: &PathBuf| std::fs::read(path).with_context(|| format!("reading {}", path.display()));
    let pin = measure_enclave::tdx_igvm::release_pin(
        &read(&cli.release_manifest)?,
        &cli.release_manifest_sha256,
        &read(&cli.igvm)?,
        &read(&cli.config)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&pin)?);
    Ok(())
}
