#![recursion_limit = "256"]
//! DeepSeek-V4.1 configuration: both JSON shapes, the per-layer table of the
//! released model, truncation, and refusals.
//!
//! The values below are those of `deepseek-ai/DeepSeek-V4.1-Flash-1017` at
//! revision `08c8657f02291e11b6f6b63e0e16014f8e748518` (`config.json` and
//! `inference/config.json`).

use eidola_engine_model::deepseek::DsConfig;
use eidola_engine_model::deepseek::config::RopeKind;
use serde_json::{Value, json};

fn compress_ratios() -> Vec<u32> {
    let mut r = vec![0, 0];
    r.extend([2; 18]);
    r.extend([1; 20]);
    r.extend([0; 3]);
    r
}

fn hf() -> Value {
    json!({
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "transformers_version": "5.6.0",
        "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
        "image_token_id": 129264,
        "quantization_config": {
            "quant_method": "fp8", "activation_scheme": "dynamic",
            "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": 129280, "hidden_size": 5120, "moe_intermediate_size": 2304,
            "num_hidden_layers": 40, "num_attention_heads": 64, "num_key_value_heads": 1,
            "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1280, "o_lora_rank": 1024,
            "o_groups": 8, "hidden_act": "silu", "swiglu_limit": 10.0, "rms_norm_eps": 1e-20,
            "attention_bias": false, "attention_dropout": 0.0, "initializer_range": 0.02,
            "use_cache": true, "tie_word_embeddings": false, "max_position_embeddings": 1048576,
            "rope_theta": 10000,
            "rope_scaling": {
                "rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
                "original_max_position_embeddings": 65536
            },
            "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6,
            "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc", "norm_topk_prob": true,
            "routed_scaling_factor": 1.5, "sliding_window": 128,
            "compress_ratios": compress_ratios(), "compress_rope_theta": 160000,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512,
            "candidate_source_layer_id": 20, "candidate_topk_blocks": 2048, "candidate_block_size": 8,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
            "engram_layer_ids": [1, 14], "engram_num_embeddings": [384006168, 384016682],
            "engram_max_ngram_size": 4, "engram_vocab_size": 16000000, "engram_n_heads": 8,
            "engram_head_dim": 256, "engram_pad_token_id": 2, "engram_compressed_vocab_size": 99092,
            "num_nextn_predict_layers": 3, "dspark_block_size": 5, "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [37, 38, 39], "dspark_markov_rank": 256,
            "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3
        },
        "vision_config": {
            "model_type": "deepseek_v41_vision", "num_hidden_layers": 32, "hidden_size": 1024,
            "num_attention_heads": 16, "intermediate_size": 2816, "patch_size": 14,
            "rope_theta": 10000, "downsample_ratio": 3, "max_image_tokens": 1024,
            "min_pixels": 295936, "max_wh_ratio": null
        }
    })
}

fn inference() -> Value {
    json!({
        "vocab_size": 129280, "dim": 5120, "moe_inter_dim": 2304, "n_layers": 40, "n_mtp_layers": 3,
        "dspark_block_size": 5, "dspark_noise_token_id": 128799, "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256, "dspark_n_routed_experts": 128, "dspark_n_activated_experts": 3,
        "n_heads": 64, "n_routed_experts": 384, "n_shared_experts": 1, "n_activated_experts": 6,
        "score_func": "sqrtsoftplus", "route_scale": 1.5, "swiglu_limit": 10.0, "q_lora_rank": 1280,
        "head_dim": 512, "rope_head_dim": 64, "norm_eps": 1e-20, "o_groups": 8, "o_lora_rank": 1024,
        "window_size": 128, "kv_source_layers": [2, 8, 14, 20],
        "index_source_layers": [2, 8, 14, 20, 24, 28, 32, 36], "original_seq_len": 65536,
        "rope_theta": 10000, "rope_factor": 16, "beta_fast": 32, "beta_slow": 1, "index_n_heads": 32,
        "index_head_dim": 128, "index_topk": 512, "candidate_source_layer": 20,
        "candidate_topk_blocks": 2048, "candidate_block_size": 8, "hc_mult": 4,
        "hc_sinkhorn_iters": 20, "hc_eps": 1e-06, "engram_layer_ids": [1, 14],
        "engram_vocab_size": 16000000, "engram_num_embeddings": [384006168, 384016682],
        "engram_max_ngram_size": 4, "engram_pad_id": 2, "engram_compressed_vocab_size": 99092,
        "dtype": "fp8", "expert_dtype": "fp4", "compress_rope_theta": 160000,
        "compress_ratios": compress_ratios(), "vision_n_layers": 32, "vision_dim": 1024,
        "vision_n_heads": 16, "vision_inter_dim": 2816, "vision_patch_size": 14,
        "vision_downsample_ratio": 3, "vision_max_n_token": 1024, "vision_min_pixels": 295936,
        "vision_max_wh_ratio": null, "image_token_id": 129264, "engram_n_heads": 8,
        "engram_head_dim": 256, "vision_rope_theta": 10000
    })
}

fn parse(v: &Value) -> eidola_engine_model::Result<DsConfig> {
    DsConfig::from_json_bytes(v.to_string().as_bytes(), "test")
}

#[test]
fn both_shapes_describe_the_same_model() {
    let (a, b) = (parse(&hf()).unwrap(), parse(&inference()).unwrap());
    assert_eq!(a.layers, b.layers);
    assert_eq!(
        (&a.attention, &a.indexer, &a.moe, &a.rope),
        (&b.attention, &b.indexer, &b.moe, &b.rope)
    );
    assert_eq!(
        (&a.engram, &a.drafter, &a.candidates),
        (&b.engram, &b.drafter, &b.candidates)
    );
    assert!(a.has_vision && b.has_vision);
    assert_eq!(a.image_token_id, Some(129264));
}

/// The sharing structure of the released model, from the config's own lists.
#[test]
fn released_layer_table() {
    let c = parse(&hf()).unwrap();
    assert_eq!(c.layers.len(), 40);
    for l in &c.layers {
        let i = l.index;
        let (ratio, kv, ix) = match i {
            0 | 1 => (0, None, None),
            2..=7 => (2, Some(2), Some(2)),
            8..=13 => (2, Some(8), Some(8)),
            14..=19 => (2, Some(14), Some(14)),
            _ => (
                1,
                Some(20),
                Some(if i < 24 { 20 } else { 24 + (i - 24) / 4 * 4 }),
            ),
        };
        assert_eq!(
            (l.compress_ratio, l.kv_source, l.index_source),
            (ratio, kv, ix),
            "layer {i}"
        );
        assert_eq!(
            l.rope,
            if ratio > 0 {
                RopeKind::Compressed
            } else {
                RopeKind::Plain
            }
        );
        assert_eq!(l.is_kv_source, [2, 8, 14, 20].contains(&i));
        assert_eq!(
            l.is_index_source,
            [2, 8, 14, 20, 24, 28, 32, 36].contains(&i)
        );
        assert_eq!(l.is_candidate_source, i == 20);
        assert_eq!(l.uses_candidates, [24, 28, 32, 36].contains(&i));
        assert_eq!(
            l.engram,
            match i {
                1 => Some(0),
                14 => Some(1),
                _ => None,
            }
        );
    }
    assert_eq!(c.indexer.weight_scale, 0.015625);
    let d = c.drafter.unwrap();
    assert_eq!(
        (
            d.blocks,
            d.slots,
            d.noise_token,
            d.moe.n_routed,
            d.moe.n_active
        ),
        (3, 5, 128799, 128, 3)
    );
    assert_eq!(d.target_layers, [37, 38, 39]);
    assert_eq!(c.engram.unwrap().lookups(), 24);
}

#[test]
fn truncation_keeps_a_valid_model_or_refuses() {
    let c = parse(&hf()).unwrap();
    let t = c
        .truncated(&[0, 1, 2, 3, 14, 15, 20, 21, 24, 37, 38, 39])
        .unwrap();
    let by_source = |s: usize| t.layers.iter().find(|l| l.source_index == s).unwrap();
    assert_eq!(by_source(3).kv_source, Some(2));
    assert_eq!(by_source(15).kv_source, Some(4));
    // Layer 37 reuses the newest kept index source over layer 20's bank.
    assert_eq!(
        (by_source(37).kv_source, by_source(37).index_source),
        (Some(6), Some(8))
    );
    assert_eq!(by_source(14).engram, Some(1));
    assert_eq!(t.drafter.as_ref().unwrap().target_layers, [9, 10, 11]);
    // A consumer without its source, and a reindexing layer without the candidate source.
    assert!(c.truncated(&[0, 3]).is_err());
    assert!(c.truncated(&[0, 21]).is_err());
    assert!(c.truncated(&[2, 1]).is_err());
    // Dropping a target layer drops the drafter, not the model.
    assert!(c.truncated(&[0, 1, 38, 39]).is_err());
    assert!(c.truncated(&[0, 1, 2]).unwrap().drafter.is_none());
    let e = t.with_routed_experts(32, 16).unwrap();
    assert_eq!((e.moe.n_routed, e.moe.source_n_routed), (32, 384));
    assert_eq!(e.drafter.as_ref().unwrap().moe.n_routed, 16);
    assert!(t.with_routed_experts(5, 16).is_err());
    let again = e.truncated(&[0, 1, 2]).unwrap();
    assert_eq!((again.moe.n_routed, again.moe.source_n_routed), (32, 384));
}

#[test]
fn unknown_keys_and_unsupported_values_are_refused() {
    let set = |path: &[&str], v: Value| {
        let mut c = hf();
        let mut at = &mut c;
        for k in &path[..path.len() - 1] {
            at = at.get_mut(*k).unwrap();
        }
        at[path[path.len() - 1]] = v;
        parse(&c)
    };
    assert!(set(&["surprise"], json!(1)).is_err());
    assert!(set(&["text_config", "surprise"], json!(1)).is_err());
    assert!(set(&["quantization_config", "surprise"], json!(1)).is_err());
    assert!(set(&["vision_config", "surprise"], json!(1)).is_err());
    assert!(set(&["text_config", "rope_scaling", "surprise"], json!(1)).is_err());
    assert!(set(&["text_config", "scoring_func"], json!("sigmoid")).is_err());
    assert!(set(&["text_config", "n_shared_experts"], json!(2)).is_err());
    assert!(set(&["quantization_config", "scale_fmt"], json!("f32")).is_err());
    assert!(set(&["text_config", "rms_norm_eps"], json!(1e-60)).is_err());
    assert!(set(&["text_config", "rope_theta"], json!(1e40)).is_err());
    assert!(set(&["text_config", "hidden_size"], json!(5100)).is_err());
    assert!(
        set(
            &["text_config", "engram_num_embeddings"],
            json!([384006168, 384016683])
        )
        .is_err()
    );
    assert!(
        set(
            &["text_config", "kv_source_layer_ids"],
            json!([1, 8, 14, 20])
        )
        .is_err()
    );
    // Fewer candidates than the top-k would leave the selection short.
    assert!(set(&["text_config", "candidate_topk_blocks"], json!(64)).is_err());
    let mut flat = inference();
    flat["surprise"] = json!(1);
    assert!(parse(&flat).is_err());
    let mut flat = inference();
    flat["gate_temp"] = json!(0.5);
    assert!(parse(&flat).is_err());
}
