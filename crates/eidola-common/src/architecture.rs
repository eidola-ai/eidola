//! The model architectures the engine knows, and the numbers each one fixes.
//!
//! A model in the catalog (`models/` at the workspace root) names one
//! [`Architecture`], and an engine deployment's `deployment.json` names the
//! architecture of the weights it serves. Everything about a deployment that
//! depends on the model's shape (KV bytes per position, the sliding window,
//! the drafter's layers, the executor's per-token scratch) is read through
//! [`Architecture::devices`], never from a free constant, so adding a model
//! family is adding a variant and the [`Shape`] of each device it runs on.
//!
//! An architecture runs on one GPU or is split across several, each device
//! holding its own layers, KV pools and weights: [`Architecture::devices`]
//! lists one [`Shape`] per device, in device order.
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
//! 3. replace the stub with its devices' [`Shape`]s, derived from the
//!    engine's own layouts, and return `true` from [`is_implemented`]
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
    /// DeepSeek-V4.1-Flash, text only: 40 layers split across two GPUs, the
    /// first 20 with the embedding on device 0 and the rest with the head and
    /// the drafter on device 1.
    DeepSeekV41Flash,
}

impl Architecture {
    /// Every architecture, implemented or not.
    pub const ALL: [Architecture; 3] = [
        Architecture::MimoV26Flash,
        Architecture::MimoV26Pro,
        Architecture::DeepSeekV41Flash,
    ];

    /// The spelling the catalog and `deployment.json` use.
    pub const fn name(self) -> &'static str {
        match self {
            Architecture::MimoV26Flash => "MimoV26Flash",
            Architecture::MimoV26Pro => "MimoV26Pro",
            Architecture::DeepSeekV41Flash => "DeepSeekV41Flash",
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
            Architecture::DeepSeekV41Flash => true,
        }
    }

    /// The numbers this architecture fixes on each device it runs on, in
    /// device order: one [`Shape`] for a model on one GPU, one per GPU for a
    /// model split across several.
    ///
    /// # Panics
    ///
    /// For an architecture that is not [implemented](Self::is_implemented).
    /// Every reader of a configuration checks that first and refuses, so
    /// reaching the panic is a missing check.
    pub fn devices(self) -> &'static [Shape] {
        match self {
            Architecture::MimoV26Flash => core::slice::from_ref(&MIMO_V26_FLASH),
            Architecture::MimoV26Pro => unimplemented!(
                "MiMo-V2.6-Pro: its KV layout, drafter, executor scratch and kernels are not \
                 implemented; nothing may name this architecture yet"
            ),
            Architecture::DeepSeekV41Flash => &DEEPSEEK_V41_FLASH,
        }
    }

    /// The numbers a one-device architecture fixes: its only device's
    /// [`Shape`].
    ///
    /// # Panics
    ///
    /// For an architecture that is not [implemented](Self::is_implemented), or
    /// one split across several devices (read [`Self::devices`] for those).
    pub fn shape(self) -> &'static Shape {
        match self.devices() {
            [shape] => shape,
            _ => panic!("{self} runs on several devices; read `Architecture::devices`"),
        }
    }

    /// The widest draft the CUDA executor can run for this architecture: the
    /// most [`Shape::mtp_layers`] any of its devices holds.
    pub fn mtp_layers(self) -> u32 {
        self.devices()
            .iter()
            .map(|d| d.mtp_layers)
            .max()
            .unwrap_or(0)
    }
}

impl core::fmt::Display for Architecture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// The numbers an architecture fixes for what a deployment allocates on one
/// device (`engine_deployment::check_resources`,
/// `engine_deployment::check_device_resources` and the bounds they are built
/// from).
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
    /// Drafter KV bytes per position whatever the draft width, for a drafter
    /// whose every block runs at any width (0 where the drafter's KV grows
    /// with the depth instead).
    pub drafter_kv_bytes_per_position: u64,
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
    /// The weights this device holds, bounded from above, for an
    /// architecture split across devices; `None` for one on a single device,
    /// whose weights are bounded by the whole pack
    /// (`engine_deployment::artifact_data_bytes`).
    pub weights_bytes: Option<u64>,
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
    drafter_kv_bytes_per_position: 0,
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
    weights_bytes: None,
};

/// One window-KV cell of DeepSeek-V4.1: 512 FP8 codes and an E8M0 scale per
/// 32 (`eidola-engine-cpu` → DeepSeek-V4.1 → Memory images).
const DS_WINDOW_CELL: u64 = 512 + 512 / 32;
/// One global-KV entry of DeepSeek-V4.1: the FP4 KV with an E4M3 scale per
/// 16, then the FP4 index key with an E8M0 scale per 32.
const DS_GLOBAL_CELL: u64 = (256 + 512 / 16) + (64 + 128 / 32);
/// Layers per device: the model splits before layer 20, its first
/// per-position KV source.
const DS_LAYERS_PER_DEVICE: u64 = 20;

/// DeepSeek-V4.1-Flash's numbers, device 0 then device 1 (the model crate's
/// `DsConfig` of the released `config.json`; the KV groups and cells of
/// `eidola-engine-cpu`'s DeepSeek executor, whose packed forms are the bytes
/// a device executor holds).
///
/// The executor-scratch fields bound a DeepSeek CUDA executor's own
/// allocations from above by the same rules as MiMo's, at this model's
/// widths. They are provisional: that executor's allocations are the
/// reference, and its derivation must be held to these numbers (or these to
/// it) when it lands.
pub const DEEPSEEK_V41_FLASH: [Shape; 2] = [
    // Device 0: the embedding, layers 0-19 with the first Engram table, and
    // layer 20's bank stub.
    Shape {
        // Three pooled KV sources (layers 2, 8, 14) at ratio 2: one entry
        // per two positions each.
        global_kv_bytes_per_position: 3 * DS_GLOBAL_CELL / 2,
        // Layers 0-19's window KV, one cell per layer.
        sliding_kv_bytes_per_position: DS_LAYERS_PER_DEVICE * DS_WINDOW_CELL,
        // `sliding_window`.
        sliding_window: 128,
        // The drafter is on device 1.
        mtp_layers: 0,
        drafter_kv_bytes_per_position_per_depth: 0,
        drafter_tap_bytes_per_depth: 0,
        drafter_kv_bytes_per_position: 0,
        // The drafter keeps no per-seat state.
        hidden_row_bytes: 0,
        step_token_device_bytes: DS_STEP_TOKEN_DEVICE_BYTES,
        expert_floor_device_bytes: DS_EXPERT_FLOOR_DEVICE_BYTES,
        sampler_row_device_bytes: DS_SAMPLER_ROW_DEVICE_BYTES,
        // Attention reads at most the window and 512 selected entries per
        // query, never a whole context: no split partial rows.
        split_partial_row_bytes: 0,
        rope_device_bytes_per_position: DS_ROPE_DEVICE_BYTES_PER_POSITION,
        // 250.86 GB in the released checkpoint's headers, rounded up.
        weights_bytes: Some(250_870_000_000),
    },
    // Device 1: layers 20-39, the norm and head, the second Engram table
    // (layer 14's), and the drafter with its own embedding copy.
    Shape {
        // Layer 20's per-position KV source.
        global_kv_bytes_per_position: DS_GLOBAL_CELL,
        // Layers 20-39's window KV.
        sliding_kv_bytes_per_position: DS_LAYERS_PER_DEVICE * DS_WINDOW_CELL,
        sliding_window: 128,
        // `num_nextn_predict_layers`; the drafter has five slots, and the
        // grammar holds a CUDA draft to `caps::CUDA_DRAFT_TOKENS`.
        mtp_layers: 3,
        drafter_kv_bytes_per_position_per_depth: 0,
        drafter_tap_bytes_per_depth: 0,
        // Every one of the drafter's three blocks keeps window KV at any
        // draft width.
        drafter_kv_bytes_per_position: 3 * DS_WINDOW_CELL,
        hidden_row_bytes: 0,
        step_token_device_bytes: DS_STEP_TOKEN_DEVICE_BYTES,
        expert_floor_device_bytes: DS_EXPERT_FLOOR_DEVICE_BYTES,
        sampler_row_device_bytes: DS_SAMPLER_ROW_DEVICE_BYTES,
        split_partial_row_bytes: 0,
        rope_device_bytes_per_position: DS_ROPE_DEVICE_BYTES_PER_POSITION,
        // 259.78 GB in the released checkpoint's headers, rounded up.
        weights_bytes: Some(259_790_000_000),
    },
];

/// The f32 logits row (129,280 × 4 bytes), the four residual copies of the
/// 5,120-wide stream, the 64 query heads of 512 and the six experts' rows:
/// under 1 MB, rounded up to 1.25 MiB.
const DS_STEP_TOKEN_DEVICE_BYTES: u64 = 5 << 18;
/// The expert layout's padding, bounded as MiMo's is: 65,536 rows of at most
/// 24 KiB.
const DS_EXPERT_FLOOR_DEVICE_BYTES: u64 = 65_536 * (24 << 10);
/// The f64 distribution over the head's 129,280 ids (1,034,240 bytes) and
/// the row's buffers, rounded up to 1.25 MiB.
const DS_SAMPLER_ROW_DEVICE_BYTES: u64 = 5 << 18;
/// 64 f32 for each of the two θ (the plain table and the compressed one).
const DS_ROPE_DEVICE_BYTES_PER_POSITION: u64 = 2 * 64 * 4;

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
            assert!(arch.mtp_layers() <= crate::engine_deployment::caps::CUDA_DRAFT_TOKENS);
        }
    }

    /// DeepSeek-V4.1-Flash on two devices, at the released checkpoint's
    /// cell sizes: 534 bytes of global KV per position on device 0 (three
    /// sources at ratio 2), 356 on device 1, 528 bytes of window KV per layer
    /// and position, the drafter's three blocks on device 1.
    #[test]
    fn deepseek_is_split_across_two_devices() {
        let ds = Architecture::DeepSeekV41Flash;
        assert!(ds.is_implemented());
        let [d0, d1] = ds.devices() else {
            panic!("two devices")
        };
        assert_eq!((DS_WINDOW_CELL, DS_GLOBAL_CELL), (528, 356));
        assert_eq!(d0.global_kv_bytes_per_position, 534);
        assert_eq!(d1.global_kv_bytes_per_position, 356);
        assert_eq!(d0.sliding_kv_bytes_per_position, 20 * 528);
        assert_eq!(d1.sliding_kv_bytes_per_position, 20 * 528);
        assert_eq!((d0.mtp_layers, d1.mtp_layers, ds.mtp_layers()), (0, 3, 3));
        assert_eq!(d1.drafter_kv_bytes_per_position, 3 * 528);
        assert_eq!(d0.weights_bytes, Some(250_870_000_000));
        assert_eq!(d1.weights_bytes, Some(259_790_000_000));
        // Every split device states its weights; a one-device architecture
        // leaves them to its pack.
        for arch in Architecture::ALL.into_iter().filter(|a| a.is_implemented()) {
            let devices = arch.devices();
            assert!(
                devices
                    .iter()
                    .all(|d| d.weights_bytes.is_some() == (devices.len() > 1)),
                "{arch}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "runs on several devices")]
    fn a_split_architecture_has_no_one_device_shape() {
        let _ = Architecture::DeepSeekV41Flash.shape();
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
