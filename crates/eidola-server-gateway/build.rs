//! Embeds the gateway's pinned trust input into the binary.
//!
//! The input, at the workspace root, is `releases/trust/engine-enclaves.json`:
//! the engine deployments this gateway build accepts, per model. It is checked
//! against the committed deployment configs under `deploy/engine/` before it
//! is embedded (`src/engine_trust/manifest.rs`): a pin that disagrees with the
//! config it names, or that names no committed config, fails the build. A TDX
//! pin's machine policy is re-derived from the vendored platform endorsements
//! its deployment names (`releases/trust/platform-endorsements/<tag>/`),
//! verified offline against `releases/trust/sigstore-trusted-root.json`, and
//! its MRTD must be the one the platform release manifest its config pins
//! states (vendored under `releases/trust/cvm-releases/`); every file read is
//! a `rerun-if-changed` input, and nothing reads the network.
//!
//! The model catalog, at the workspace root, is `models/<family>/<label>.toml`:
//! one file per set of weights, keyed by its weights hash. Each file is
//! checked (`eidola_common::model_catalog`, the reader the client's build runs too), the files are checked together (no hash
//! twice, each tag naming exactly one hash), and the rows are bound to the
//! engine pins one to one by hash before they are rendered as Rust.

#[path = "src/engine_trust/manifest.rs"]
mod engine_manifest;
// `engine_manifest`'s `super::file`: the typed parse the runtime reads with.
#[path = "src/engine_trust/file.rs"]
mod file;
use eidola_common::model_catalog as models;

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("workspace root above crates/eidola-server-gateway")
        .to_path_buf();

    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("src/engine_trust/manifest.rs").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("src/engine_trust/file.rs").display()
    );
    let (engine_enclaves, pinned) = engine_enclaves(&workspace_root);
    fs::write(out_dir.join("engine_enclaves.gen.rs"), engine_enclaves)
        .expect("write engine_enclaves.gen.rs");
    let catalog = model_catalog(&workspace_root, &pinned);
    fs::write(out_dir.join("model_catalog.gen.rs"), catalog).expect("write model_catalog.gen.rs");
}

/// Every catalog file under `models/`, read, checked, bound to the engine
/// pins and rendered as Rust.
fn model_catalog(workspace_root: &Path, pinned: &[engine_manifest::CheckedModel]) -> String {
    let root = workspace_root.join(models::MODELS_ROOT);
    println!("cargo:rerun-if-changed={}", root.display());
    let mut rows = Vec::new();
    let entries = |dir: &Path| -> Vec<fs::DirEntry> {
        let Ok(read) = fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut list: Vec<fs::DirEntry> = read
            .map(|e| e.unwrap_or_else(|e| panic!("list {}: {e}", dir.display())))
            .collect();
        list.sort_by_key(|e| e.file_name());
        list
    };
    let utf8 = |e: &fs::DirEntry| {
        e.file_name()
            .into_string()
            .unwrap_or_else(|n| panic!("{n:?} under models/ is not UTF-8"))
    };
    for family in entries(&root) {
        let family_name = utf8(&family);
        if !family.path().is_dir() {
            // The directory's README is for people; nothing else sits here.
            assert!(
                family_name == "README.md",
                "models/{family_name}: catalog files live at models/<family>/<label>.toml"
            );
            continue;
        }
        println!("cargo:rerun-if-changed={}", family.path().display());
        for file in entries(&family.path()) {
            let relative = format!("{}/{family_name}/{}", models::MODELS_ROOT, utf8(&file));
            println!("cargo:rerun-if-changed={}", file.path().display());
            let bytes = fs::read(file.path()).unwrap_or_else(|e| panic!("read {relative}: {e}"));
            rows.push(
                models::parse(&relative, &bytes)
                    .unwrap_or_else(|e| panic!("the model catalog is not valid: {e}")),
            );
        }
    }
    models::check_catalog(&rows).unwrap_or_else(|e| panic!("the model catalog is not valid: {e}"));
    let pins: Vec<models::PinView<'_>> = pinned
        .iter()
        .map(|m| models::PinView {
            sha256: &m.id,
            architecture: m.architecture,
            repo: &m.weights.repo,
            revision: &m.weights.revision,
        })
        .collect();
    models::check_binding(&rows, &pins)
        .unwrap_or_else(|e| panic!("the model catalog and the engine pins disagree: {e}"));
    let windows: Vec<(&str, &str, u64)> = rows
        .iter()
        .map(|r| (r.path.as_str(), r.sha256.as_str(), r.context_length))
        .collect();
    engine_manifest::check_context_windows(&windows, pinned)
        .unwrap_or_else(|e| panic!("the model catalog and the engine pins disagree: {e}"));
    render_catalog(&rows)
}

/// The checked rows as a Rust constant.
fn render_catalog(rows: &[models::Row]) -> String {
    let modalities = |list: &[String]| -> String {
        let items: Vec<String> = list
            .iter()
            .map(|m| match m.as_str() {
                "text" => "Modality::Text".to_string(),
                "image" => "Modality::Image".to_string(),
                "audio" => "Modality::Audio".to_string(),
                other => unreachable!("checked modality {other}"),
            })
            .collect();
        format!("&[{}]", items.join(", "))
    };
    let mut out = String::new();
    out.push_str("// @generated by build.rs from models/ — do not edit.\n");
    out.push_str("pub const MODEL_CATALOG: &[CatalogRow] = &[\n");
    for row in rows {
        let price = match row.price {
            models::Price::PerToken {
                input_per_m,
                output_per_m,
            } => format!(
                "RowPrice::PerToken {{ input_per_m: {input_per_m:?}, output_per_m: {output_per_m:?}, \
                 cached_input_per_m: {:?} }}",
                row.cached_input_per_m
            ),
            models::Price::PerRequest { usd } => format!("RowPrice::PerRequest {{ usd: {usd:?} }}"),
        };
        let lifecycle = match &row.lifecycle {
            models::Lifecycle::Current => "Lifecycle::Current".to_string(),
            models::Lifecycle::Deprecated { retire_after } => {
                format!("Lifecycle::Deprecated {{ retire_after: {retire_after:?} }}")
            }
            models::Lifecycle::Retired { retire_after } => {
                format!("Lifecycle::Retired {{ retire_after: {retire_after:?} }}")
            }
        };
        let files: Vec<String> = row
            .files
            .iter()
            .map(|f| {
                format!(
                    "WeightsFile {{ path: {:?}, size: {}, sha256: {:?} }}",
                    f.path, f.size, f.sha256
                )
            })
            .collect();
        let class = match row.output_budget_class.as_str() {
            "standard" => "OutputBudgetClass::Standard",
            "reasoning" => "OutputBudgetClass::Reasoning",
            other => unreachable!("checked budget class {other}"),
        };
        writeln!(
            out,
            "    CatalogRow {{ sha256: {:?}, architecture: Architecture::{}, tags: &{:?}, \
             repo: {:?}, revision: {:?}, files: &[{}], name: {:?}, description: {:?}, \
             context_length: {}, tool_calling: {}, reasoning: {}, input_modalities: {}, \
             output_modalities: {}, max_output_tokens: {:?}, output_budget_class: {}, \
             price: {}, lifecycle: {} }},",
            row.sha256,
            row.architecture.name(),
            row.tags,
            row.repo,
            row.revision,
            files.join(", "),
            row.name,
            row.description,
            row.context_length,
            row.tool_calling,
            row.reasoning,
            modalities(&row.input_modalities),
            modalities(&row.output_modalities),
            row.max_output_tokens,
            class,
            price,
            lifecycle,
        )
        .unwrap();
    }
    out.push_str("];\n");
    out
}

/// The config path of every `deploy/engine/<name>/<variant>/` directory.
/// Every directory counts, whatever it holds, so a deployment missing its
/// config is reported rather than overlooked.
fn committed_deployments(workspace_root: &Path) -> Vec<String> {
    let root = workspace_root.join(engine_manifest::DEPLOY_ROOT);
    println!("cargo:rerun-if-changed={}", root.display());
    let dirs = |dir: &Path| -> Vec<String> {
        let Ok(entries) = fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .map(|e| e.unwrap_or_else(|e| panic!("list {}: {e}", dir.display())))
            .filter(|e| e.path().is_dir())
            .map(|e| {
                e.file_name()
                    .into_string()
                    .unwrap_or_else(|n| panic!("{n:?} under {} is not UTF-8", dir.display()))
            })
            .collect();
        names.sort();
        names
    };
    let mut committed = Vec::new();
    for model in dirs(&root) {
        for variant in dirs(&root.join(&model)) {
            committed.push(format!(
                "{}/{model}/{variant}/tinfoil-config.yml",
                engine_manifest::DEPLOY_ROOT
            ));
        }
    }
    committed
}

/// Check `releases/trust/engine-enclaves.json` against the deployments it
/// names and render the pinned models as Rust.
fn engine_enclaves(workspace_root: &Path) -> (String, Vec<engine_manifest::CheckedModel>) {
    let path = workspace_root.join("releases/trust/engine-enclaves.json");
    println!("cargo:rerun-if-changed={}", path.display());
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut read = |relative: &str| {
        let file = workspace_root.join(relative);
        println!("cargo:rerun-if-changed={}", file.display());
        fs::read(&file).map_err(|e| format!("read {relative}: {e}"))
    };
    let pinned = engine_manifest::check(&bytes, &mut read)
        .unwrap_or_else(|e| panic!("releases/trust/engine-enclaves.json is not valid: {e}"));
    let committed = committed_deployments(workspace_root);
    engine_manifest::require_every_deployment_pinned(&committed, &pinned)
        .unwrap_or_else(|e| panic!("releases/trust/engine-enclaves.json is not valid: {e}"));

    let mut out = String::new();
    out.push_str("// @generated by build.rs — do not edit.\n");
    writeln!(
        out,
        "pub const ENGINE_ENCLAVES_JSON: &str = include_str!({:?});",
        path
    )
    .unwrap();
    out.push_str("pub const PINNED_MODELS: &[PinnedModel] = &[\n");
    for model in &pinned {
        writeln!(
            out,
            "    PinnedModel {{ id: {:?}, architecture: \
             eidola_common::architecture::Architecture::{}, weights: PinnedWeights {{ sha256: \
             {:?}, repo: {:?}, revision: {:?} }}, prompt_cache: PromptCachePolicy {{ enabled: {}, \
             idle_ttl_secs: {}, max_age_secs: {} }}, deployments: &{:?} }},",
            model.id,
            model.architecture.name(),
            model.weights.sha256,
            model.weights.repo,
            model.weights.revision,
            model.prompt_cache.enabled,
            model.prompt_cache.idle_ttl_secs,
            model.prompt_cache.max_age_secs,
            model.deployments,
        )
        .unwrap();
    }
    out.push_str("];\n");
    (out, pinned)
}
