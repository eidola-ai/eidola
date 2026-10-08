//! The one model configuration this executor supports.
//!
//! The executor supports exactly the configuration space a real checkpoint
//! exercises and the tests cover: MiMo-V2.6-Flash's values, field for field,
//! with no permitted-but-untested degrees of freedom. Shapes are template
//! arguments or compile-time constants in the kernels, so another shape would
//! read or write out of bounds rather than fail; scalars the kernels take at run
//! time (RoPE θ, epsilon, value scale, routing scale, window) are held to the
//! values the tests run too. [`check_supported`] refuses everything else before
//! anything is allocated or launched. The `ModelConfig` parser accepts more of
//! the MiMo family than this; Pro differs in hidden size (6144), query heads
//! (128), global KV heads (8), experts (384), value scale and epsilon, and is the
//! scope of the multi-GPU executor, not a widening of this one.

use std::fmt;

use eidola_engine_model::ModelConfig;
use eidola_engine_model::config::{AttentionKind, FfnKind};

/// A configuration field the kernels cannot run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported {
    /// The field, e.g. `layers[3].attention.head_dim_qk`.
    pub field: String,
    pub found: String,
    pub required: String,
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} is {}, the kernels require {}",
            self.field, self.found, self.required
        )
    }
}

impl std::error::Error for Unsupported {}

/// Hidden size: DeepGEMM's instances (N and K of both projections) and the
/// UE8M0 quantization's 512-wide words.
pub const HIDDEN: usize = 4096;
/// Head rows: the BF16 `lm_head` GEMM.
pub const VOCAB: usize = 152_576;
/// Dense (layer 0) intermediate size: the FP8 GEMMs and the f32-scale SwiGLU.
pub const DENSE_INTER: usize = 16_384;
/// QK head dim: the FA2 instances and `eidola_qkv_rope_kv`.
pub const HEAD_DIM_QK: usize = 192;
/// V head dim: the FA2 instances, the merge kernel and `eidola_qkv_rope_kv`.
pub const HEAD_DIM_V: usize = 128;
/// Rotated dims per Q/K head: `eidola_qkv_rope_kv` and the RoPE tables.
pub const ROPE_DIM: usize = 64;
/// Query heads, every layer: the attention plan is made once per KV group.
pub const Q_HEADS: usize = 64;
/// KV heads of global layers.
pub const GLOBAL_KV_HEADS: usize = 4;
/// KV heads of sliding layers.
pub const SLIDING_KV_HEADS: usize = 8;
/// Visible positions of sliding layers.
pub const WINDOW: usize = 128;
/// Routed experts: DeepGEMM's group count, `eidola_router_topk`'s and
/// `eidola_moe_permute`'s per-expert shared arrays.
pub const EXPERTS: usize = 256;
/// Expert intermediate size: DeepGEMM's instances.
pub const EXPERT_INTER: usize = 2048;
/// Experts per token, exactly: the router kernel always renormalizes the
/// selection (the reference skips that for a single expert).
pub const TOP_K: usize = 8;
/// Most experts per token any launch wrapper accepts: the router's selection
/// array.
pub const MAX_TOP_K: usize = 8;
/// Layers of the checkpoint: the range `keep_layers` selects from.
pub const SOURCE_LAYERS: usize = 48;
/// Rank-major chunks of the fused QKV (`tp_size`, or the top-level
/// `num_key_value_heads` when the checkpoint does not record it): the
/// loader's de-interleaving.
pub const QKV_CHUNKS: usize = 4;
/// FP8 scale tiles: the FP8 GEMMs' per-128 scale grids.
pub const FP8_BLOCK: [usize; 2] = [128, 128];
/// Elements per E8M0 scale of the MXFP4 experts: DeepGEMM's SFB words.
pub const MXFP4_BLOCK: usize = 32;
/// Flash's global-attention layers (`hybrid_layer_pattern` 0); every other
/// layer is sliding-window.
pub const GLOBAL_LAYERS: [usize; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];
/// Flash's dense-FFN layers (`moe_layer_freq` 0); every other layer routes
/// to experts.
pub const DENSE_LAYERS: [usize; 1] = [0];
/// Largest sampleable vocabulary: `sampling.cu`'s 1,024 chunks of 1,024.
pub const MAX_SAMPLEABLE: usize = 1 << 20;

/// Refuse every configuration but MiMo-V2.6-Flash's (any subset of its layers).
pub fn check_supported(c: &ModelConfig) -> Result<(), Unsupported> {
    fn bad<T>(
        field: impl Into<String>,
        found: impl fmt::Debug,
        required: impl fmt::Debug,
    ) -> Result<T, Unsupported> {
        Err(Unsupported {
            field: field.into(),
            found: format!("{found:?}"),
            required: format!("{required:?}"),
        })
    }
    macro_rules! exact {
        ($field:expr, $found:expr, $want:expr) => {
            if $found != $want {
                return bad($field, $found, $want);
            }
        };
    }
    exact!("hidden_size", c.hidden_size, HIDDEN);
    exact!("vocab_size", c.vocab_size, VOCAB);
    exact!(
        "dense_intermediate_size",
        c.dense_intermediate_size,
        DENSE_INTER
    );
    exact!(
        "max_position_embeddings",
        c.max_position_embeddings,
        MAX_POSITIONS
    );
    exact!("rms_norm_eps", c.rms_norm_eps, 1e-6f32);
    exact!("source_num_layers", c.source_num_layers, SOURCE_LAYERS);
    // Read by no implementation; pinned so a checkpoint declaring another
    // value is refused rather than served as if it agreed.
    exact!("attention_chunk_size", c.attention_chunk_size, Some(WINDOW));
    // Read only as the fused-QKV chunk count's fallback (`qkv_chunks`), and
    // so invisible to every per-layer check: pinned here.
    exact!("num_key_value_heads", c.num_key_value_heads, QKV_CHUNKS);
    let Some(q) = &c.quant else {
        return bad("quant", "None", "FP8 blocks with MXFP4 experts");
    };
    exact!("quant.fp8_block", q.fp8_block, FP8_BLOCK);
    exact!("quant.mxfp4_block", q.mxfp4_block, Some(MXFP4_BLOCK));
    // Activations are quantized per row at run time (`amax / 448`); a
    // static or absent scheme would mean another contract.
    exact!(
        "quant.activation_scheme",
        q.activation_scheme.as_deref(),
        Some("dynamic")
    );
    // BF16 exactly where the loader reads BF16: every `o_proj` (the
    // checkpoint's layers and its MTP decoder) and nothing else.
    let mut want: Vec<String> = (0..SOURCE_LAYERS)
        .map(|i| format!("model.layers.{i}.self_attn.o_proj"))
        .chain(std::iter::once("model.decoder.self_attn.o_proj".into()))
        .collect();
    let mut found = q.ignored_layers.clone();
    want.sort();
    found.sort();
    if found != want {
        return bad(
            "quant.ignored_layers",
            found.len(),
            "every o_proj, and nothing else",
        );
    }
    exact!(
        "attention_value_scale",
        c.attention_value_scale,
        Some(0.707f32)
    );
    for (i, l) in c.layers.iter().enumerate() {
        // Each retained layer is Flash's layer of that index, kind for kind.
        let global = GLOBAL_LAYERS.contains(&l.source_index);
        exact!(
            format!("layers[{i}].attention.kind"),
            matches!(l.attention.kind, AttentionKind::Global),
            global
        );
        exact!(
            format!("layers[{i}].ffn"),
            l.ffn,
            if DENSE_LAYERS.contains(&l.source_index) {
                FfnKind::Dense
            } else {
                FfnKind::Moe
            }
        );
        let a = &l.attention;
        let f = |name: &str| format!("layers[{i}].attention.{name}");
        exact!(f("head_dim_qk"), a.head_dim_qk, HEAD_DIM_QK);
        exact!(f("head_dim_v"), a.head_dim_v, HEAD_DIM_V);
        exact!(f("rope_dim"), a.rope_dim, ROPE_DIM);
        exact!(f("num_q_heads"), a.num_q_heads, Q_HEADS);
        match a.kind {
            AttentionKind::Global => {
                exact!(f("num_kv_heads"), a.num_kv_heads, GLOBAL_KV_HEADS);
                exact!(f("rope_theta"), a.rope_theta, 1e7f32);
                exact!(f("has_sinks"), a.has_sinks, false);
            }
            AttentionKind::Sliding { window } => {
                exact!(f("window"), window, WINDOW);
                exact!(f("num_kv_heads"), a.num_kv_heads, SLIDING_KV_HEADS);
                exact!(f("rope_theta"), a.rope_theta, 1e4f32);
                exact!(f("has_sinks"), a.has_sinks, true);
            }
        }
    }
    let Some(m) = &c.moe else {
        return bad("moe", "None", "the routed-expert block");
    };
    exact!("moe.num_experts", m.num_experts, EXPERTS);
    exact!("moe.intermediate_size", m.intermediate_size, EXPERT_INTER);
    exact!("moe.top_k", m.top_k, TOP_K);
    exact!("moe.norm_topk_prob", m.norm_topk_prob, true);
    exact!("moe.routed_scaling_factor", m.routed_scaling_factor, 1.0f32);
    // The loader reads the router weight as BF16 and the router kernel
    // scores it in f32: Flash's declaration, and no other.
    exact!(
        "moe.router_dtype",
        m.router_dtype.as_deref(),
        Some("bfloat16")
    );
    Ok(())
}

/// The checkpoint's context window (`max_position_embeddings`).
pub const MAX_POSITIONS: usize = 1 << 20;

/// Shared memory per block the largest launch needs: the CUTLASS BF16 GEMM's
/// dynamic plus static bytes. Every
/// kernel's dynamic plus static shared memory is checked against this on a
/// device (`tests/smoke.rs`), and its maximum must equal it.
pub const REQUIRED_SMEM_PER_BLOCK: u32 = 231_424;

/// Refuse a device the retained kernels cannot run on, for this model:
/// returns the image it runs (`forced`, if given and compatible).
///
/// - a compute capability with a built image (10.x), and a forced image that
///   matches it (`sm_100a` only on 10.0, `sm_103a` only on 10.3);
/// - opt-in shared memory for the largest launch ([`REQUIRED_SMEM_PER_BLOCK`]);
/// - with any MoE layer: at least the SM count DeepGEMM's persistent instances
///   are built for, and cluster launches (their 2-CTA clusters).
///
/// The launch paths keep their own checks (`ops::deepgemm_grid`, the
/// shared-memory opt-in at load); this runs before any device memory is used.
pub fn check_device(
    info: &crate::DeviceInfo,
    config: &ModelConfig,
    forced: Option<crate::ImageArch>,
) -> Result<crate::ImageArch, Unsupported> {
    use crate::ImageArch;
    let bad = |field: &str, found: String, required: String| Unsupported {
        field: format!("device.{field}"),
        found,
        required,
    };
    let (major, minor) = info.compute_capability;
    let native = ImageArch::for_compute_capability(major, minor).ok_or_else(|| {
        bad(
            "compute_capability",
            format!("{major}.{minor}"),
            "10.x".into(),
        )
    })?;
    let image = match forced {
        None => native,
        Some(ImageArch::Sm100f) => ImageArch::Sm100f,
        Some(f) if f == native => f,
        Some(f) => {
            return Err(bad(
                "compute_capability",
                format!("{major}.{minor}"),
                format!("the forced image {}'s exact part", f.as_str()),
            ));
        }
    };
    if info.max_smem_per_block_optin < REQUIRED_SMEM_PER_BLOCK {
        return Err(bad(
            "max_smem_per_block_optin",
            info.max_smem_per_block_optin.to_string(),
            format!("at least {REQUIRED_SMEM_PER_BLOCK}"),
        ));
    }
    let moe = config
        .layers
        .iter()
        .any(|l| l.ffn == eidola_engine_model::config::FfnKind::Moe);
    if moe {
        let sms = crate::ops::DEEPGEMM_INSTANCE_SMS;
        if info.sm_count < sms {
            return Err(bad(
                "sm_count",
                info.sm_count.to_string(),
                format!("at least {sms} (DeepGEMM's persistent grid)"),
            ));
        }
        if !info.cluster_launch {
            return Err(bad(
                "cluster_launch",
                "false".into(),
                "true (DeepGEMM's 2-CTA clusters)".into(),
            ));
        }
    }
    Ok(image)
}

/// The sampleable vocabulary the sampler can serve with this head.
/// Refuse a layer selection that is not a strictly ascending subset of the
/// checkpoint's layers: the retained model keeps the checkpoint's order, and
/// no layer twice.
pub fn check_layers(keep: &[usize], source_layers: usize) -> Result<(), Unsupported> {
    let ascending = keep.windows(2).all(|w| w[0] < w[1]);
    if keep.is_empty() || !ascending || keep.last().is_some_and(|&l| l >= source_layers) {
        return Err(Unsupported {
            field: "keep_layers".into(),
            found: format!("{keep:?}"),
            required: format!("a strictly ascending, non-empty subset of 0..{source_layers}"),
        });
    }
    Ok(())
}

/// The fused QKV's rank-major chunk count: the checkpoint's `tp_size`
/// metadata, else the configuration's `num_key_value_heads`; either way
/// Flash's [`QKV_CHUNKS`].
pub fn qkv_chunks(tp_size: Option<&str>, c: &ModelConfig) -> Result<usize, Unsupported> {
    let found = match tp_size {
        Some(s) => s.trim().parse().ok(),
        None => Some(c.num_key_value_heads),
    };
    match found {
        Some(QKV_CHUNKS) => Ok(QKV_CHUNKS),
        _ => Err(Unsupported {
            field: if tp_size.is_some() {
                "tp_size"
            } else {
                "num_key_value_heads"
            }
            .into(),
            found: tp_size.map_or(c.num_key_value_heads.to_string(), |s| format!("{s:?}")),
            required: QKV_CHUNKS.to_string(),
        }),
    }
}

/// Refuse a context longer than the checkpoint's declared window: positions
/// past `max_position_embeddings` are outside what the model was trained for.
pub fn check_context(max_model_len: u32, c: &ModelConfig) -> Result<(), Unsupported> {
    if max_model_len as usize > c.max_position_embeddings {
        return Err(Unsupported {
            field: "max_model_len".into(),
            found: max_model_len.to_string(),
            required: format!("at most {}", c.max_position_embeddings),
        });
    }
    Ok(())
}

pub fn check_sampleable(sampleable: usize, vocab: usize) -> Result<(), Unsupported> {
    if sampleable == 0 || sampleable > vocab || sampleable > MAX_SAMPLEABLE {
        return Err(Unsupported {
            field: "sampleable_vocab_size".into(),
            found: sampleable.to_string(),
            required: format!("1..={}", vocab.min(MAX_SAMPLEABLE)),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_engine_model::config::{AttentionSpec, MoeSpec};

    fn load(name: &str) -> ModelConfig {
        let path = format!(
            "{}/../eidola-engine-model/tests/data/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        ModelConfig::from_file(std::path::Path::new(&path)).unwrap()
    }

    fn flash() -> ModelConfig {
        load("flash-mopd.config.json")
    }

    fn refused(c: &ModelConfig, field: &str) {
        let e = check_supported(c).expect_err(field);
        assert_eq!(e.field, field, "{e}");
    }

    #[test]
    fn flash_and_its_truncations_are_supported() {
        check_supported(&flash()).unwrap();
        check_supported(&flash().truncated(&[0, 1, 2, 5]).unwrap()).unwrap();
        check_sampleable(151_675, 152_576).unwrap();
        check_context(1 << 20, &flash()).unwrap();
    }

    /// A context past the checkpoint's window is refused; the window itself
    /// is a pinned value (`every_field_is_checked`).
    #[test]
    fn contexts_are_bounded() {
        let e = check_context((1 << 20) + 1, &flash()).expect_err("past the window");
        assert_eq!(e.field, "max_model_len");
    }

    fn b300() -> crate::DeviceInfo {
        crate::DeviceInfo {
            ordinal: 0,
            name: "test".into(),
            compute_capability: (10, 3),
            sm_count: 148,
            driver_version: 13000,
            total_mem_bytes: 287_428_640_768,
            max_smem_per_block_optin: 232_448,
            cluster_launch: true,
        }
    }

    /// Every device-side requirement, missed alone, is refused by name.
    #[test]
    fn devices_are_checked() {
        use crate::ImageArch;
        let c = flash();
        assert_eq!(check_device(&b300(), &c, None).unwrap(), ImageArch::Sm103a);
        assert_eq!(
            check_device(&b300(), &c, Some(ImageArch::Sm100f)).unwrap(),
            ImageArch::Sm100f
        );
        type D = fn(&mut crate::DeviceInfo);
        for (field, set) in [
            ("device.sm_count", (|d| d.sm_count = 132) as D),
            ("device.max_smem_per_block_optin", |d| {
                d.max_smem_per_block_optin = REQUIRED_SMEM_PER_BLOCK - 1
            }),
            ("device.cluster_launch", |d| d.cluster_launch = false),
            ("device.compute_capability", |d| {
                d.compute_capability = (12, 0)
            }),
            ("device.compute_capability", |d| {
                d.compute_capability = (9, 0)
            }),
        ] {
            let mut d = b300();
            set(&mut d);
            assert_eq!(check_device(&d, &c, None).expect_err(field).field, field);
        }
        let e = check_device(&b300(), &c, Some(ImageArch::Sm100a)).expect_err("sm_100a on 10.3");
        assert_eq!(e.field, "device.compute_capability");
        // Without an MoE layer, DeepGEMM's SM count and clusters do not apply.
        let dense = c.truncated(&[0]).unwrap();
        let mut d = b300();
        d.sm_count = 132;
        d.cluster_launch = false;
        check_device(&d, &dense, None).unwrap();
    }

    /// Only a strictly ascending subset of the checkpoint's layers is kept.
    #[test]
    fn layer_selections_are_checked() {
        for keep in [&[0, 1, 2, 5][..], &[1, 5], &[47], &[0]] {
            check_layers(keep, SOURCE_LAYERS).unwrap();
        }
        for keep in [&[][..], &[1, 0], &[0, 0], &[0, 5, 2], &[48], &[0, 48]] {
            let e = check_layers(keep, SOURCE_LAYERS).expect_err("refused");
            assert_eq!(e.field, "keep_layers", "{keep:?}");
        }
    }

    /// The fused QKV is read as Flash's four chunks, whichever source names
    /// the count.
    #[test]
    fn qkv_chunks_are_flash_s() {
        let c = flash();
        assert_eq!(qkv_chunks(None, &c).unwrap(), QKV_CHUNKS);
        assert_eq!(qkv_chunks(Some(" 4 "), &c).unwrap(), QKV_CHUNKS);
        for tp in ["2", "8", "four", ""] {
            assert_eq!(qkv_chunks(Some(tp), &c).unwrap_err().field, "tp_size");
        }
        let mut c = flash().truncated(&[1, 2]).unwrap();
        c.num_key_value_heads = 2;
        assert_eq!(
            qkv_chunks(None, &c).unwrap_err().field,
            "num_key_value_heads"
        );
    }

    #[test]
    fn pro_is_refused() {
        refused(&load("pro-mopd.config.json"), "hidden_size");
    }

    /// Every field, changed alone, is refused with that field named.
    #[test]
    fn every_field_is_checked() {
        type C = fn(&mut ModelConfig);
        for (name, set) in [
            ("hidden_size", (|c| c.hidden_size = 6144) as C),
            ("vocab_size", |c| c.vocab_size = 152_584),
            ("dense_intermediate_size", |c| {
                c.dense_intermediate_size = 16_512
            }),
            ("max_position_embeddings", |c| {
                c.max_position_embeddings = 262_144
            }),
            ("rms_norm_eps", |c| c.rms_norm_eps = 1e-5),
            ("attention_value_scale", |c| {
                c.attention_value_scale = Some(0.612)
            }),
            ("attention_value_scale", |c| c.attention_value_scale = None),
            ("moe", |c| c.moe = None),
        ] {
            let mut c = flash();
            set(&mut c);
            refused(&c, name);
        }
        // Layer 0 is global, layer 1 sliding.
        type A = fn(&mut AttentionSpec);
        for (layer, name, set) in [
            (0, "head_dim_qk", (|a| a.head_dim_qk = 128) as A),
            (1, "head_dim_qk", |a| a.head_dim_qk = 128),
            (1, "head_dim_v", |a| a.head_dim_v = 192),
            (0, "rope_dim", |a| a.rope_dim = 96),
            (0, "num_q_heads", |a| a.num_q_heads = 32),
            (1, "num_q_heads", |a| a.num_q_heads = 128),
            (0, "num_kv_heads", |a| a.num_kv_heads = 8),
            (1, "num_kv_heads", |a| a.num_kv_heads = 4),
            (0, "rope_theta", |a| a.rope_theta = 1e4),
            (1, "rope_theta", |a| a.rope_theta = 1e7),
            (0, "has_sinks", |a| a.has_sinks = true),
            (1, "has_sinks", |a| a.has_sinks = false),
            (0, "kind", |a| {
                a.kind = eidola_engine_model::config::AttentionKind::Sliding { window: 128 }
            }),
            (1, "kind", |a| {
                a.kind = eidola_engine_model::config::AttentionKind::Global
            }),
            (1, "window", |a| {
                a.kind = eidola_engine_model::config::AttentionKind::Sliding { window: 256 }
            }),
        ] {
            let mut c = flash();
            set(&mut c.layers[layer].attention);
            refused(&c, &format!("layers[{layer}].attention.{name}"));
        }
        // Each layer's FFN kind is Flash's for that layer, in a truncation too.
        for (keep, i, ffn) in [
            (&[0, 1][..], 0, FfnKind::Moe),
            (&[0, 1][..], 1, FfnKind::Dense),
            (&[5, 6][..], 1, FfnKind::Dense),
        ] {
            let mut c = flash().truncated(keep).unwrap();
            c.layers[i].ffn = ffn;
            refused(&c, &format!("layers[{i}].ffn"));
        }
        // And its attention kind, wherever it sits in the selection.
        let mut c = flash().truncated(&[6, 11]).unwrap();
        c.layers[1].attention = c.layers[0].attention.clone();
        refused(&c, "layers[1].attention.kind");
        type M = fn(&mut MoeSpec);
        for (name, set) in [
            ("moe.num_experts", (|m| m.num_experts = 384) as M),
            ("moe.intermediate_size", |m| m.intermediate_size = 1024),
            ("moe.top_k", |m| m.top_k = 1),
            ("moe.top_k", |m| m.top_k = 7),
            ("moe.norm_topk_prob", |m| m.norm_topk_prob = false),
            ("moe.routed_scaling_factor", |m| {
                m.routed_scaling_factor = 2.5
            }),
            ("moe.router_dtype", |m| {
                m.router_dtype = Some("float32".into())
            }),
            ("moe.router_dtype", |m| m.router_dtype = None),
        ] {
            let mut c = flash();
            set(c.moe.as_mut().unwrap());
            refused(&c, name);
        }
        for (name, set) in [
            ("source_num_layers", (|c| c.source_num_layers = 47) as C),
            ("num_key_value_heads", |c| c.num_key_value_heads = 2),
            ("attention_chunk_size", |c| {
                c.attention_chunk_size = Some(256)
            }),
            ("attention_chunk_size", |c| c.attention_chunk_size = None),
            ("quant", |c| c.quant = None),
            ("quant.fp8_block", |c| {
                c.quant.as_mut().unwrap().fp8_block = [64, 64]
            }),
            ("quant.mxfp4_block", |c| {
                c.quant.as_mut().unwrap().mxfp4_block = Some(16)
            }),
            ("quant.activation_scheme", |c| {
                c.quant.as_mut().unwrap().activation_scheme = Some("static".into())
            }),
            ("quant.activation_scheme", |c| {
                c.quant.as_mut().unwrap().activation_scheme = None
            }),
            ("quant.ignored_layers", |c| {
                c.quant.as_mut().unwrap().ignored_layers.pop();
            }),
            ("quant.ignored_layers", |c| {
                let q = c.quant.as_mut().unwrap();
                q.ignored_layers[0] = "model.layers.0.self_attn.qkv_proj".into();
            }),
        ] {
            let mut c = flash();
            set(&mut c);
            refused(&c, name);
            // Sliding-only subsets read none of these per layer: still refused.
            let mut c = flash().truncated(&[1, 2]).unwrap();
            set(&mut c);
            refused(&c, name);
        }
        for (s, v) in [(0, 152_576), (152_577, 152_576), ((1 << 20) + 1, 2 << 20)] {
            assert!(check_sampleable(s, v).is_err(), "{s} of {v}");
        }
    }
}
