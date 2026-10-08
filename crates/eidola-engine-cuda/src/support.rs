//! The one model configuration this executor supports.
//!
//! The kernels are built for one shape family. Shapes are template arguments
//! or compile-time constants in them, not runtime parameters, so anything
//! else would read or write out of bounds rather than fail. [`check_supported`]
//! refuses every other configuration before anything is allocated or launched.
//! Each requirement below names the kernel that bakes it in; the
//! `ModelConfig` parser accepts more of the MiMo family than this.

use std::fmt;

use eidola_engine_model::ModelConfig;

use crate::moe_gemm::GROUPS;

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
/// QK head dim: the FA2 instances and `eidola_qkv_rope_kv`.
pub const HEAD_DIM_QK: usize = 192;
/// V head dim: the FA2 instances, the merge kernel and `eidola_qkv_rope_kv`.
pub const HEAD_DIM_V: usize = 128;
/// Rotated dims per Q/K head: `eidola_qkv_rope_kv` and the RoPE tables.
pub const ROPE_DIM: usize = 64;
/// Routed experts: DeepGEMM's group count, `eidola_router_topk`'s and
/// `eidola_moe_permute`'s per-expert shared arrays.
pub const EXPERTS: usize = GROUPS as usize;
/// Expert intermediate size: DeepGEMM's instances.
pub const EXPERT_INTER: usize = 2048;
/// Most experts per token: `eidola_router_topk`'s selection array.
pub const MAX_TOP_K: usize = 8;
/// Largest sampleable vocabulary: `sampling.cu`'s 1,024 chunks of 1,024.
pub const MAX_SAMPLEABLE: usize = 1 << 20;

/// Refuse every configuration the kernels cannot run.
pub fn check_supported(c: &ModelConfig) -> Result<(), Unsupported> {
    let bad = |field: String, found: String, required: &str| {
        Err(Unsupported {
            field,
            found,
            required: required.to_string(),
        })
    };
    if c.hidden_size != HIDDEN {
        return bad("hidden_size".into(), c.hidden_size.to_string(), "4096");
    }
    // FP8 blockwise GEMM tiles (128) and the f32-scale SwiGLU quantization.
    if !c.dense_intermediate_size.is_multiple_of(128) {
        return bad(
            "dense_intermediate_size".into(),
            c.dense_intermediate_size.to_string(),
            "a multiple of 128",
        );
    }
    // The BF16 lm_head GEMM's N alignment.
    if !c.vocab_size.is_multiple_of(8) {
        return bad(
            "vocab_size".into(),
            c.vocab_size.to_string(),
            "a multiple of 8",
        );
    }
    for (i, l) in c.layers.iter().enumerate() {
        let a = &l.attention;
        let f = |name: &str| format!("layers[{i}].attention.{name}");
        if a.head_dim_qk != HEAD_DIM_QK {
            return bad(f("head_dim_qk"), a.head_dim_qk.to_string(), "192");
        }
        if a.head_dim_v != HEAD_DIM_V {
            return bad(f("head_dim_v"), a.head_dim_v.to_string(), "128");
        }
        if a.rope_dim != ROPE_DIM {
            return bad(f("rope_dim"), a.rope_dim.to_string(), "64");
        }
        if a.num_kv_heads == 0 || !a.num_q_heads.is_multiple_of(a.num_kv_heads) {
            return bad(
                f("num_q_heads"),
                format!("{} (with {} KV heads)", a.num_q_heads, a.num_kv_heads),
                "a multiple of the KV head count",
            );
        }
    }
    if let Some(m) = &c.moe {
        if m.num_experts != EXPERTS {
            return bad("moe.num_experts".into(), m.num_experts.to_string(), "256");
        }
        if m.intermediate_size != EXPERT_INTER {
            return bad(
                "moe.intermediate_size".into(),
                m.intermediate_size.to_string(),
                "2048",
            );
        }
        if m.top_k == 0 || m.top_k > MAX_TOP_K {
            return bad("moe.top_k".into(), m.top_k.to_string(), "1..=8");
        }
        // `eidola_router_topk` always renormalizes the selected scores.
        if !m.norm_topk_prob {
            return bad("moe.norm_topk_prob".into(), "false".into(), "true");
        }
    }
    Ok(())
}

/// The sampleable vocabulary the sampler can serve with this head.
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

    fn flash() -> ModelConfig {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../eidola-engine-model/tests/data/flash-mopd.config.json"
        );
        ModelConfig::from_file(std::path::Path::new(path)).unwrap()
    }

    fn refused(c: &ModelConfig, field: &str) {
        let e = check_supported(c).expect_err(field);
        assert_eq!(e.field, field, "{e}");
    }

    #[test]
    fn the_flash_checkpoint_is_supported() {
        check_supported(&flash()).unwrap();
        check_sampleable(151_675, 152_576).unwrap();
    }

    /// Every field the kernels bake in, changed one at a time, is refused
    /// with that field named.
    #[test]
    fn every_baked_in_field_is_checked() {
        let mut c = flash();
        c.hidden_size = 6144;
        refused(&c, "hidden_size");

        let mut c = flash();
        c.dense_intermediate_size = 16_400;
        refused(&c, "dense_intermediate_size");

        let mut c = flash();
        c.vocab_size = 152_577;
        refused(&c, "vocab_size");

        for (name, set) in [
            (
                "head_dim_qk",
                (|a| a.head_dim_qk = 128) as fn(&mut eidola_engine_model::config::AttentionSpec),
            ),
            ("head_dim_v", |a| a.head_dim_v = 192),
            ("rope_dim", |a| a.rope_dim = 96),
            ("num_q_heads", |a| a.num_q_heads = 63),
        ] {
            for layer in [0, 7] {
                let mut c = flash();
                set(&mut c.layers[layer].attention);
                refused(&c, &format!("layers[{layer}].attention.{name}"));
            }
        }

        type Set = fn(&mut eidola_engine_model::config::MoeSpec);
        for (name, set) in [
            ("moe.num_experts", (|m| m.num_experts = 384) as Set),
            ("moe.intermediate_size", |m| m.intermediate_size = 1024),
            ("moe.top_k", |m| m.top_k = 9),
            ("moe.top_k", |m| m.top_k = 0),
            ("moe.norm_topk_prob", |m| m.norm_topk_prob = false),
        ] {
            let mut c = flash();
            set(c.moe.as_mut().unwrap());
            refused(&c, name);
        }

        for (s, v) in [(0, 152_576), (152_577, 152_576), ((1 << 20) + 1, 2 << 20)] {
            assert!(check_sampleable(s, v).is_err(), "{s} of {v}");
        }
    }
}
