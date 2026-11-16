//! MiMo-V2.6 (Flash-MOPD and Pro-MOPD): configuration, weight loading and the
//! dense reference forward.
//!
//! - [`config`] parses `config.json` into a per-layer table and rejects
//!   anything the reference does not implement.
//! - [`dequant`] turns FP8 block-scaled, MXFP4 and rank-major pre-sharded
//!   fused-QKV storage into exact f32.
//! - [`weights`] assembles a model (whole, or a subset of layers) from a
//!   weight set.
//! - [`attention`] is RoPE, window visibility and single-query attention with
//!   a sink.
//! - [`forward`] is the reference forward: main model and the MTP draft
//!   layers.
//!
//! See `AGENTS.md` in this crate for what is verified against what.

pub mod attention;
pub mod config;
pub mod dequant;
pub mod forward;
pub mod weights;

use std::path::Path;
use std::sync::Arc;

pub use config::ModelConfig;
pub use forward::{ForwardOptions, ForwardOutput, LogitsAt, MtpOutput, ReferenceModel};
pub use weights::{LoadOptions, ModelWeights};

use crate::error::{Error, Result};
use crate::safetensors::WeightSet;

/// The model configuration from the `config.json` a weight set read (and
/// hashed) in [`WeightSet::open_dir`].
pub fn config_of(store: &WeightSet) -> Result<ModelConfig> {
    let bytes = store
        .semantic_file("config.json")
        .ok_or_else(|| Error::Safetensors {
            path: store.dir().join("config.json"),
            reason: "no config.json was read with these weights".into(),
        })?;
    ModelConfig::from_json_bytes(bytes, "config.json")
}

/// Load `config.json` and every safetensors file in `dir`, optionally keeping
/// only the listed checkpoint layers. With `manifest` (file name → sha256), every
/// file the load reads, `config.json` and the index included, must match it.
pub fn load_reference(
    dir: &Path,
    keep_layers: Option<&[usize]>,
    opts: &LoadOptions,
    manifest: Option<&std::collections::BTreeMap<String, String>>,
) -> Result<ReferenceModel> {
    let store = WeightSet::open_dir(dir)?;
    if let Some(m) = manifest {
        store.verify_sha256(m)?;
    }
    let mut config = config_of(&store)?;
    if let Some(keep) = keep_layers {
        config = config.truncated(keep)?;
    }
    let store = Arc::new(store);
    Ok(ReferenceModel::new(ModelWeights::load(
        store, config, opts,
    )?))
}
