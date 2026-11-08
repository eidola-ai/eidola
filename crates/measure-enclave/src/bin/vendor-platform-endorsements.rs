//! Vendor one release of the platform provider's signed platform
//! endorsements into `releases/trust/platform-endorsements/<tag>/`.
//!
//! Usage (the `just engine-platform-endorsements <tag>` recipe downloads the
//! two inputs into an empty temporary directory and runs this):
//!   cargo run -p measure-enclave --bin vendor-platform-endorsements -- \
//!     --workspace . \
//!     --tag platform-vX.Y.Z \
//!     --endorsements <downloaded platform-endorsements.json> \
//!     --attestations <GitHub's attestations response for its sha256>
//!
//! Nothing here reads the network. The downloaded file and the one bundle
//! among the attestations that verifies it as `--tag` (offline, against the
//! committed Sigstore trusted root, exactly as the gateway's build will) are
//! written; the files written are then read back and verified again. Any
//! failure writes nothing it has not verified.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Vendor a verified platform endorsements release")]
struct Cli {
    /// The workspace root.
    #[arg(long, default_value = ".")]
    workspace: PathBuf,
    /// The release tag, `platform-vX.Y.Z`.
    #[arg(long)]
    tag: String,
    /// The release's `platform-endorsements.json`, as downloaded.
    #[arg(long)]
    endorsements: PathBuf,
    /// GitHub's `attestations/sha256:<hex>` response for that file.
    #[arg(long)]
    attestations: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let read =
        |path: &PathBuf| std::fs::read(path).with_context(|| format!("reading {}", path.display()));
    let vendored = measure_enclave::platform_endorsements::vendor(
        &cli.workspace,
        &cli.tag,
        &read(&cli.endorsements)?,
        &read(&cli.attestations)?,
    )?;
    println!(
        "verified {} (Rekor log index {})",
        cli.tag, vendored.rekor_log_index
    );
    println!("wrote {}", vendored.files.0);
    println!("wrote {}", vendored.files.1);
    println!("sha256 {}", vendored.sha256);
    println!("TDX policies (machines):");
    for (name, machines) in &vendored.tdx_policies {
        println!("  {name} ({machines})");
    }
    println!(
        "A deployment adopts it in its deployment.json:\n  \"platform_endorsements\": \
         {{\"release\": \"{}\", \"sha256\": \"{}\", \"policy\": \"<one of the above>\"}}",
        cli.tag, vendored.sha256
    );
    Ok(())
}
