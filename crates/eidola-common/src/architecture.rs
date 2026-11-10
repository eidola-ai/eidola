//! The model architectures the engine knows, and the numbers each one fixes.
//!
//! A model in the catalog (`models/` at the workspace root) names one
//! [`Architecture`], and an engine deployment's `deployment.json` names the
//! architecture of the weights it serves. Everything about a deployment that
//! depends on the model's shape (KV bytes per position, the sliding window,
//! the drafter's layers, the executor's per-token scratch) is read through
//! [`Architecture::shape`], never from a free constant, so adding a model
//! family is adding a variant and its [`Shape`].
//!
//! An architecture is **implemented** when the engine can serve it: its
//! shapes are known here and the engine crates load and run it. A variant
//! that is not implemented is a deliberate stub: [`Architecture::shape`]
//! panics for it, and the gateway's build refuses any catalog row or engine
//! deployment that names it, so a stub can be written down before its engine
//! work lands without anything being able to serve it.
//!
//! Adding a model family:
//!
//! 1. add the variant, with its [`name`](Architecture::name) (the spelling
//!    the catalog and `deployment.json` use) and a stub `shape` arm;
//! 2. implement it in the engine crates (`eidola-engine-model`'s config and
//!    loader, the CUDA executor's support check and kernels);
//! 3. replace the stub with its [`Shape`], derived from the engine's own
//!    layouts, and return `true` from [`is_implemented`]
//!    (Architecture::is_implemented).

/// A model architecture, as the catalog and deployments name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Architecture {
    /// MiMo-V2.6-Flash: 9 global and 39 sliding attention layers, three MTP
    /// layers, the architecture the engine serves.
    MimoV26Flash,
    /// MiMo-V2.6-Pro. **Not implemented**: a stub, so the pattern for adding
    /// a model family is in the tree. Its shapes, loader and kernels do not
    /// exist yet, and nothing may name it in a catalog row or a deployment.
    MimoV26Pro,
}

impl Architecture {
    /// Every architecture, implemented or not.
    pub const ALL: [Architecture; 2] = [Architecture::MimoV26Flash, Architecture::MimoV26Pro];

    /// The spelling the catalog and `deployment.json` use.
    pub const fn name(self) -> &'static str {
        match self {
            Architecture::MimoV26Flash => "MimoV26Flash",
            Architecture::MimoV26Pro => "MimoV26Pro",
        }
    }

    /// The architecture spelled `name`, if there is one.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.name() == name)
    }

    /// Whether the engine can serve this architecture: its [`Shape`] is
    /// known and the engine crates run it.
    pub const fn is_implemented(self) -> bool {
        match self {
            Architecture::MimoV26Flash => true,
            Architecture::MimoV26Pro => false,
        }
    }

    /// The numbers this architecture fixes.
    ///
    /// # Panics
    ///
    /// For an architecture that is not [implemented](Self::is_implemented).
    /// Every reader of a configuration checks that first and refuses, so
    /// reaching the panic is a missing check.
    pub fn shape(self) -> &'static Shape {
        match self {
            Architecture::MimoV26Flash => &MIMO_V26_FLASH,
            Architecture::MimoV26Pro => unimplemented!(
                "MiMo-V2.6-Pro: its KV layout, drafter, executor scratch and kernels are not \
                 implemented; nothing may name this architecture yet"
            ),
        }
    }
}

impl core::fmt::Display for Architecture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// The numbers an architecture fixes for what a deployment allocates
/// (`engine_deployment::check_resources` and the bounds it is built from).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// KV-cache bytes per position of the global attention layers.
    pub global_kv_bytes_per_position: u64,
    /// KV-cache bytes per position of the sliding attention layers.
    pub sliding_kv_bytes_per_position: u64,
    /// Positions a sliding layer sees.
    pub sliding_window: u64,
    /// The MTP layers the checkpoint ships, each serving one draft depth:
    /// the widest draft the CUDA executor can run.
    pub mtp_layers: u32,
    /// Drafter KV bytes per position for each draft depth (one MTP layer).
    pub drafter_kv_bytes_per_position_per_depth: u64,
    /// Drafter boundary-tap bytes per block for each draft depth: one chain
    /// level of the hidden size, in f32.
    pub drafter_tap_bytes_per_depth: u64,
    /// Bytes of one row of the hidden size in f32 (the drafter's per-seat
    /// state rows).
    pub hidden_row_bytes: u64,
    /// Device scratch per token a step may hold (`MAX_BATCHED_TOKENS`).
    pub step_token_device_bytes: u64,
    /// The expert scratch held whatever the step.
    pub expert_floor_device_bytes: u64,
    /// Device memory per sampler row.
    pub sampler_row_device_bytes: u64,
    /// Device bytes of one split-attention partial row.
    pub split_partial_row_bytes: u64,
    /// RoPE table bytes per position (`MAX_MODEL_LEN`).
    pub rope_device_bytes_per_position: u64,
}

impl Shape {
    /// KV-cache bytes per position, every layer.
    pub const fn kv_bytes_per_position(&self) -> u64 {
        self.global_kv_bytes_per_position + self.sliding_kv_bytes_per_position
    }
}

/// MiMo-V2.6-Flash's numbers (`eidola-engine-cuda` `support`, `kv`,
/// `model::ScratchSizes`, `attention`, the sampler).
pub const MIMO_V26_FLASH: Shape = Shape {
    // 9 global layers of 4 KV heads, each head 192 K + 128 V dims, bf16.
    global_kv_bytes_per_position: 9 * 4 * (192 + 128) * 2,
    // 39 sliding layers of 8 KV heads, each head 192 K + 128 V dims, bf16.
    sliding_kv_bytes_per_position: 39 * 8 * (192 + 128) * 2,
    // `support::WINDOW`.
    sliding_window: 128,
    // `num_nextn_predict_layers`, `support::MTP_LAYERS`.
    mtp_layers: 3,
    // One MTP layer: 8 KV heads of 192 K + 128 V dims, bf16
    // (`kv::GroupGeometry`, planar layout).
    drafter_kv_bytes_per_position_per_depth: 8 * (192 + 128) * 2,
    // One chain level of the hidden size (4,096) in f32.
    drafter_tap_bytes_per_depth: 4096 * 4,
    hidden_row_bytes: 4096 * 4,
    // The f32 logits row (152,576 × 4 bytes), the residual stream and its
    // projections, the quantized activations, Q, K, V and attention output,
    // the dense FFN, and eight expert rows of 22,580 bytes each, about
    // 1.03 MB in all, rounded up to 1.25 MiB.
    step_token_device_bytes: 5 << 18,
    // The expert layout's padding (up to 127 rows for each of the 256
    // experts, and a block's rounding), bounded by 65,536 rows of at most
    // 24 KiB.
    expert_floor_device_bytes: 65_536 * (24 << 10),
    // The f64 distribution over at most the head's 152,576 ids (1,220,608
    // bytes) and the row's buffers, rounded up to 1.25 MiB.
    sampler_row_device_bytes: 5 << 18,
    // 64 query heads, each a BF16 output of 128 and an f32 log-sum-exp
    // (`attention::PARTIAL_BYTES_PER_HEAD`).
    split_partial_row_bytes: 64 * (128 * 2 + 4),
    // 64 f32 for each distinct θ, at most three.
    rope_device_bytes_per_position: 3 * 64 * 4,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_architecture_round_trips_its_name() {
        for arch in Architecture::ALL {
            assert_eq!(Architecture::from_name(arch.name()), Some(arch));
        }
        assert_eq!(Architecture::from_name("mimo-v2.6-flash"), None);
        assert_eq!(Architecture::from_name(""), None);
    }

    #[test]
    fn flash_is_the_implemented_architecture() {
        assert!(Architecture::MimoV26Flash.is_implemented());
        let flash = Architecture::MimoV26Flash.shape();
        assert_eq!(
            flash.kv_bytes_per_position(),
            (9 * 4 + 39 * 8) * (192 + 128) * 2
        );
        assert_eq!(flash.mtp_layers, 3);
    }

    #[test]
    fn every_implemented_architecture_drafts_within_the_nodes_cap() {
        for arch in Architecture::ALL.into_iter().filter(|a| a.is_implemented()) {
            assert!(arch.shape().mtp_layers <= crate::engine_deployment::caps::CUDA_DRAFT_TOKENS);
        }
    }

    /// The stub stays a stub until its engine work lands: asking for its
    /// shape is a missing `is_implemented` check, and panics.
    #[test]
    #[should_panic(expected = "MiMo-V2.6-Pro")]
    fn the_pro_stub_has_no_shape() {
        assert!(!Architecture::MimoV26Pro.is_implemented());
        let _ = Architecture::MimoV26Pro.shape();
    }
}
