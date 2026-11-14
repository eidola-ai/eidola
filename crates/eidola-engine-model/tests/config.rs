//! The published MiMo-V2.6 Flash-MOPD and Pro-MOPD `config.json` files
//! (`tests/data/`, copied verbatim from the model repositories) parse into the
//! layer tables the architecture calls for.

use std::path::Path;

use eidola_engine_model::Error;
use eidola_engine_model::config::{AttentionKind, FfnKind, ModelConfig};
use serde_json::Value;

fn data(name: &str) -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}

fn parse(v: &Value) -> Result<ModelConfig, Error> {
    ModelConfig::from_json_bytes(&serde_json::to_vec(v).unwrap(), "test")
}

#[test]
fn flash_layer_table() {
    let c = parse(&data("flash-mopd.config.json")).unwrap();
    assert_eq!(c.hidden_size, 4096);
    assert_eq!(c.num_layers(), 48);
    assert_eq!(
        c.global_layer_indices(),
        vec![0, 5, 11, 17, 23, 29, 35, 41, 47]
    );
    assert_eq!(c.vocab_size, 152_576);
    assert_eq!(c.rms_norm_eps, 1e-6);
    assert_eq!(c.attention_value_scale, Some(0.707));
    assert_eq!(c.num_key_value_heads, 4);

    let ga = &c.layers[0].attention;
    assert_eq!(ga.kind, AttentionKind::Global);
    assert_eq!((ga.num_q_heads, ga.num_kv_heads), (64, 4));
    assert_eq!((ga.head_dim_qk, ga.head_dim_v, ga.rope_dim), (192, 128, 64));
    assert_eq!(ga.rope_theta, 1e7);
    assert!(!ga.has_sinks);
    assert_eq!(ga.softmax_scale, 192f32.powf(-0.5));
    assert_eq!(ga.qkv_rows(), 4 * 3392);

    let swa = &c.layers[1].attention;
    assert_eq!(swa.kind, AttentionKind::Sliding { window: 128 });
    assert_eq!((swa.num_q_heads, swa.num_kv_heads), (64, 8));
    assert_eq!(swa.rope_dim, 64);
    assert_eq!(swa.rope_theta, 1e4);
    assert!(swa.has_sinks);
    assert_eq!(swa.qkv_rows(), 4 * 3712);

    assert_eq!(c.layers[0].ffn, FfnKind::Dense);
    assert!(c.layers[1..].iter().all(|l| l.ffn == FfnKind::Moe));
    assert_eq!(c.dense_intermediate_size, 16384);
    let moe = c.moe.as_ref().unwrap();
    assert_eq!(
        (moe.num_experts, moe.top_k, moe.intermediate_size),
        (256, 8, 2048)
    );
    assert!(moe.norm_topk_prob);
    assert_eq!(moe.routed_scaling_factor, 1.0);

    assert_eq!(c.mtp.declared_layers, Some(3));
    assert_eq!(c.mtp.attention, *swa);
    assert_eq!(c.mtp.intermediate_size, 16384);
    let q = c.quant.as_ref().unwrap();
    assert_eq!(q.fp8_block, [128, 128]);
    assert_eq!(q.mxfp4_block, Some(32));
    assert_eq!(q.activation_scheme.as_deref(), Some("dynamic"));
    assert_eq!(moe.router_dtype.as_deref(), Some("bfloat16"));
    assert_eq!(c.attention_chunk_size, Some(128));
}

#[test]
fn pro_layer_table() {
    let c = parse(&data("pro-mopd.config.json")).unwrap();
    assert_eq!(c.hidden_size, 6144);
    assert_eq!(c.num_layers(), 70);
    assert_eq!(
        c.global_layer_indices(),
        vec![0, 7, 15, 23, 31, 39, 47, 55, 62, 69]
    );
    assert_eq!(c.rms_norm_eps, 1e-5);
    assert_eq!(c.attention_value_scale, Some(0.612));
    let ga = &c.layers[0].attention;
    assert_eq!((ga.num_q_heads, ga.num_kv_heads), (128, 8));
    let swa = &c.layers[1].attention;
    assert_eq!((swa.num_q_heads, swa.num_kv_heads), (128, 8));
    assert_eq!((ga.rope_theta, swa.rope_theta), (1e7, 1e4));
    assert_eq!(c.moe.as_ref().unwrap().num_experts, 384);
    // Pro ships MTP weights but leaves the count null.
    assert_eq!(c.mtp.declared_layers, None);
}

#[test]
fn truncation_keeps_source_indices() {
    let c = parse(&data("flash-mopd.config.json")).unwrap();
    let t = c.truncated(&[0, 1, 2, 5]).unwrap();
    assert_eq!(t.num_layers(), 4);
    assert_eq!(t.global_layer_indices(), vec![0, 5]);
    assert_eq!(t.layers[3].index, 3);
    assert_eq!(t.layers[3].source_index, 5);
    assert_eq!(t.layers[3].ffn, FfnKind::Moe);
    assert!(c.truncated(&[0, 0]).is_err());
    assert!(c.truncated(&[48]).is_err());
}

fn rejects(field: &str, value: Value) {
    let mut v = data("flash-mopd.config.json");
    v[field] = value.clone();
    match parse(&v) {
        Err(Error::UnsupportedConfig(_)) | Err(Error::InvalidConfig(_)) => {}
        other => panic!("{field} = {value}: expected rejection, got {other:?}"),
    }
}

#[test]
fn unsupported_features_are_rejected() {
    rejects("model_type", "mimo_v2_flash".into());
    rejects("hidden_act", "gelu".into());
    rejects("attention_bias", true.into());
    rejects("tie_word_embeddings", true.into());
    rejects("attention_projection_layout", "split".into());
    rejects("scoring_func", "softmax".into());
    rejects("topk_method", "greedy".into());
    rejects("n_group", 8.into());
    rejects("n_shared_experts", 1.into());
    rejects("hybrid_block_size", 4.into());
    rejects(
        "rope_scaling",
        serde_json::json!({"rope_type": "yarn", "factor": 4.0}),
    );
    rejects("partial_rotary_factor", 0.5.into());
    rejects("sliding_window_size", 256.into());
    rejects("moe_router_dtype", "float16".into());
    rejects("swa_rope_theta", 0.into());

    let mut v = data("flash-mopd.config.json");
    v["hybrid_layer_pattern"].as_array_mut().unwrap().pop();
    assert!(parse(&v).is_err());

    let mut v = data("flash-mopd.config.json");
    v["quantization_config"]["weight_block_size"] = serde_json::json!(null);
    assert!(parse(&v).is_err());

    let mut v = data("flash-mopd.config.json");
    v["quantization_config"]["mxfp4_block_size"] = 16.into();
    assert!(parse(&v).is_err());

    // An unknown quantization key is refused rather than dropped.
    let mut v = data("flash-mopd.config.json");
    v["quantization_config"]["activation_block_size"] = 128.into();
    assert!(parse(&v).is_err());

    let mut v = data("flash-mopd.config.json");
    v["rope_parameters"]["rope_type"] = "yarn".into();
    assert!(parse(&v).is_err());
}

/// Float fields are validated as the forward stores them (f32), not as JSON doubles:
/// a value that is fine in f64 but overflows or underflows in f32 is refused.
#[test]
fn float_fields_are_validated_after_conversion_to_f32() {
    let base = data("flash-mopd.config.json");
    let refused = |edit: &dyn Fn(&mut Value)| {
        let mut v = base.clone();
        edit(&mut v);
        matches!(parse(&v), Err(Error::InvalidConfig(_)))
    };
    // rope_theta: 1e40 is finite in f64 and infinite in f32.
    assert!(refused(&|v| {
        v["rope_theta"] = 1e40.into();
        v["rope_parameters"]["rope_theta"] = 1e40.into();
    }));
    assert!(refused(&|v| v["swa_rope_theta"] = 1e40.into()));
    assert!(refused(&|v| v["swa_rope_theta"] = 1e-50.into()));
    assert!(refused(&|v| v["layernorm_epsilon"] = 1e-50.into()));
    assert!(refused(&|v| v["layernorm_epsilon"] = 0.0.into()));
    assert!(refused(&|v| v["attention_value_scale"] = 1e39.into()));
    assert!(refused(&|v| v["attention_value_scale"] = (-0.5).into()));
    assert!(refused(&|v| v["routed_scaling_factor"] = 1e39.into()));
    assert!(refused(&|v| v["routed_scaling_factor"] = 0.0.into()));
    // The published values still parse.
    assert!(parse(&base).is_ok());
}
