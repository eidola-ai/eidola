//! Qwen3.8 and Qwen3.5 (text): configuration, weight loading and the dense reference
//! forward.
//!
//! - [`config`] reads `config.json` into a typed per-layer table.
//! - [`numeric`], [`linear`], [`deltanet`] and [`attention`] are the kernels,
//!   each with its rounding points and summation orders fixed. An executor
//!   that wants bit equality with this oracle calls them.
//! - [`weights`] opens a model from a weight set, its large weights left in
//!   the mapped checkpoint.
//! - [`forward`] is the reference forward, one layer at a time over any
//!   contiguous range of positions, and the MTP draft layer.
//!
//! See `AGENTS.md` in this crate for the numerics contract and what is
//! verified against what.

pub mod attention;
pub mod config;
pub mod deltanet;
pub mod forward;
pub mod linear;
pub mod numeric;
pub mod weights;

use std::path::Path;
use std::sync::Arc;

pub use config::{LayerKind, QwenConfig, WeightFormat};
pub use deltanet::DeltaNetState;
pub use forward::{
    KvCache, LayerState, LogitRows, MtpOutput, QwenForwardOptions, QwenForwardOutput, QwenModel,
    SequenceState,
};
pub use weights::QwenWeights;

use crate::error::{Error, Result};
use crate::safetensors::WeightSet;

/// The configuration a weight directory's `config.json` describes.
pub fn config_of(store: &WeightSet) -> Result<QwenConfig> {
    let bytes = store
        .semantic_file("config.json")
        .ok_or_else(|| Error::Safetensors {
            path: store.dir().join("config.json"),
            reason: "no config.json was read with these weights".into(),
        })?;
    QwenConfig::from_json_bytes(bytes, "config.json")
}

/// Open `config.json` and every safetensors file in `dir`. `configure` may
/// truncate the configuration before the weights are opened.
pub fn load_reference(
    dir: &Path,
    configure: impl FnOnce(QwenConfig) -> Result<QwenConfig>,
) -> Result<QwenModel> {
    let store = Arc::new(WeightSet::open_dir(dir)?);
    let config = configure(config_of(&store)?)?;
    Ok(QwenModel::new(QwenWeights::load(store, config)?))
}
