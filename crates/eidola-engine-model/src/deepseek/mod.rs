//! DeepSeek-V4.1 (text): configuration, weight loading and the dense
//! reference forward, beside the MiMo-V2.6 modules at the crate root.
//!
//! - [`config`] reads either config shape into a typed per-layer table.
//! - [`numeric`], [`quant`], [`linear`], [`rope`], [`mhc`], [`attention`],
//!   [`moe`] and [`engram`] are the kernels, each with its rounding points
//!   and summation orders fixed. An executor that wants bit equality with
//!   this oracle calls them.
//! - [`weights`] assembles a model from a weight set.
//! - [`forward`] is the reference forward, one layer at a time over any
//!   contiguous range of positions; [`drafter`] is the draft head.
//!
//! See `AGENTS.md` in this crate for the numerics contract and what is
//! verified against what.

pub mod attention;
pub mod config;
pub mod drafter;
pub mod engram;
pub mod forward;
pub mod linear;
pub mod mhc;
pub mod moe;
pub mod numeric;
pub mod quant;
pub mod rope;
pub mod weights;

use std::path::Path;
use std::sync::Arc;

pub use config::DsConfig;
pub use forward::{
    Boundary, DsForwardOptions, DsForwardOutput, DsModel, GlobalBank, LayerOptions, Rows,
    SequenceState,
};
pub use weights::{DsLoadOptions, DsWeights};

use crate::error::{Error, Result};
use crate::safetensors::WeightSet;

/// The configuration a weight directory's `config.json` describes.
pub fn config_of(store: &WeightSet) -> Result<DsConfig> {
    let bytes = store
        .semantic_file("config.json")
        .ok_or_else(|| Error::Safetensors {
            path: store.dir().join("config.json"),
            reason: "no config.json was read with these weights".into(),
        })?;
    DsConfig::from_json_bytes(bytes, "config.json")
}

/// Load `config.json` and every safetensors file in `dir`. `configure` may
/// truncate the configuration before the weights are read. `token_map` is the
/// Engram compressed-vocabulary map.
pub fn load_reference(
    dir: &Path,
    configure: impl FnOnce(DsConfig) -> Result<DsConfig>,
    opts: &DsLoadOptions,
    token_map: Option<Vec<u32>>,
) -> Result<DsModel> {
    let store = Arc::new(WeightSet::open_dir(dir)?);
    let config = configure(config_of(&store)?)?;
    DsModel::new(DsWeights::load(store, config, opts)?, token_map)
}
