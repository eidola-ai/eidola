//! MiMo-V2.6 model definition: configuration, weight loading, and the dense
//! f32 reference forward pass.
//!
//! - [`config`] parses `config.json` into a per-layer table and rejects
//!   anything the reference does not implement.
//! - [`safetensors`] memory-maps checkpoint shards with lazy per-tensor access
//!   and optional sha256 verification.
//! - [`dequant`] turns FP8 block-scaled, MXFP4 and rank-major pre-sharded
//!   fused-QKV storage into exact f32.
//! - [`weights`] assembles a model (whole, or a subset of layers) from a
//!   weight set.
//! - [`forward`] is the reference forward: main model and the MTP draft
//!   layers.
//! - [`compare`] holds the metrics golden checks report.
//!
//! See `AGENTS.md` in this crate for what is verified against what.

pub mod attention;
pub mod compare;
pub mod config;
pub mod dequant;
mod error;
pub mod forward;
pub mod numeric;
pub mod safetensors;
pub mod tensor;
pub mod weights;

use std::path::Path;
use std::sync::Arc;

pub use config::ModelConfig;
pub use error::{Error, Result};
pub use forward::{ForwardOptions, ForwardOutput, LogitsAt, MtpOutput, ReferenceModel};
pub use safetensors::WeightSet;
pub use tensor::Matrix;
pub use weights::{LoadOptions, ModelWeights};

/// Load `config.json` and every safetensors file in `dir`, optionally keeping
/// only the listed checkpoint layers.
pub fn load_reference(
    dir: &Path,
    keep_layers: Option<&[usize]>,
    opts: &LoadOptions,
) -> Result<ReferenceModel> {
    let mut config = ModelConfig::from_file(&dir.join("config.json"))?;
    if let Some(keep) = keep_layers {
        config = config.truncated(keep)?;
    }
    let store = Arc::new(WeightSet::open_dir(dir)?);
    Ok(ReferenceModel::new(ModelWeights::load(
        store, config, opts,
    )?))
}
