//! Compute `releases/trust/engine-enclaves.json` from the committed engine
//! deployments under `deploy/engine/`.
//!
//! Usage:
//!   cargo run -p measure-enclave --bin measure-engine-enclaves -- \
//!     --workspace . \
//!     --release-dir <dir> \
//!     --output releases/trust/engine-enclaves.json
//!
//! `--release-dir` holds, for every `cvm-version` a deployment selects, the
//! platform provider's `tinfoil-inference-v<version>-manifest.json` and
//! `tinfoil-tdx-v<version>.igvm`. The manifest must hash to the pin in the
//! config's `cvm-version`, and the image to the manifest's record of it.
//! Release-time work, like `just update-manifest`: nothing here downloads.
//! With no deployments committed it writes the empty pin set.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use measure_enclave::engine_enclaves::{self, DeploymentInputs};

#[derive(Parser)]
#[command(about = "Compute engine-enclaves.json from the committed engine deployments")]
struct Cli {
    /// The workspace root.
    #[arg(long, default_value = ".")]
    workspace: PathBuf,
    /// Directory holding each selected release's manifest and TDX IGVM image.
    #[arg(long)]
    release_dir: PathBuf,
    /// Where to write the file (stdout when absent).
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let read =
        |path: &Path| std::fs::read(path).with_context(|| format!("reading {}", path.display()));
    let root = cli.workspace.join("deploy/engine");

    let mut entries: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
    for model in sorted_dirs(&root)? {
        let model_id = name(&model)?;
        for variant in sorted_dirs(&model)? {
            let variant_name = name(&variant)?;
            let config_path = format!("deploy/engine/{model_id}/{variant_name}/tinfoil-config.yml");
            let config = read(&variant.join("tinfoil-config.yml"))?;
            let sidecar = read(&variant.join("deployment.json"))?;
            let (version, _) =
                engine_enclaves::required_release(&config).with_context(|| config_path.clone())?;
            let release_manifest = read(
                &cli.release_dir
                    .join(format!("tinfoil-inference-v{version}-manifest.json")),
            )?;
            let igvm = read(&cli.release_dir.join(format!("tinfoil-tdx-v{version}.igvm")))?;
            let entry = engine_enclaves::deployment_entry(
                &model_id,
                &DeploymentInputs {
                    config_path: &config_path,
                    config: &config,
                    sidecar: &sidecar,
                    release_manifest: &release_manifest,
                    igvm: &igvm,
                },
            )
            .with_context(|| config_path.clone())?;
            entries.entry(model_id.clone()).or_default().push(entry);
        }
    }

    let rendered = engine_enclaves::render(entries)?;
    match cli.output {
        Some(path) => std::fs::write(&path, rendered)
            .with_context(|| format!("writing {}", path.display()))?,
        None => print!("{rendered}"),
    }
    Ok(())
}

/// The subdirectories of `dir`, sorted; none when `dir` does not exist.
fn sorted_dirs(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

fn name(path: &Path) -> Result<String> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("{} has no UTF-8 name", path.display()))?;
    // The component grammar the gateway's build holds a pinned config's path
    // to: a directory outside it is refused here, before anything is measured.
    ensure!(
        eidola_common::engine_deployment::is_safe_component(name),
        "{} is not a deployment path component ([a-z0-9][a-z0-9._-]*)",
        path.display()
    );
    Ok(name.to_owned())
}
