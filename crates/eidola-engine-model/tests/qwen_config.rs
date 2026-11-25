//! Qwen3.8 and Qwen3.5 configuration: the released `config.json`s, their
//! per-layer tables, truncation, and refusals.
//!
//! `tests/data/qwen3.8-27b.config.json` is the file of `Qwen/Qwen3.8-27B-FP8`
//! at revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`,
//! `tests/data/qwen3.5-2b.config.json` that of `Qwen/Qwen3.5-2B` at
//! `15852e8c16360a2fea060d615a32b45270f8a8fc`, and
//! `tests/data/qwen3.5-0.8b.config.json` that of `Qwen/Qwen3.5-0.8B` at
//! `2fc06364715b967f1860aea9cf38778875588b17`, byte for byte (each SHA-256 is
//! checked against the recorded one below).

use eidola_engine_model::qwen::config::LayerKind;
use eidola_engine_model::qwen::{QwenConfig, WeightFormat};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const RELEASED: &[u8] = include_bytes!("data/qwen3.8-27b.config.json");
const RECORDED: &str = include_str!("../golden/qwen/upstream-sha256.json");
const QWEN35_2B: &[u8] = include_bytes!("data/qwen3.5-2b.config.json");
const QWEN35_2B_RECORDED: &str = include_str!("../golden/qwen/upstream-qwen3.5-2b-sha256.json");
const QWEN35_08B: &[u8] = include_bytes!("data/qwen3.5-0.8b.config.json");
const QWEN35_08B_RECORDED: &str = include_str!("../golden/qwen/upstream-qwen3.5-0.8b-sha256.json");

fn released() -> Value {
    serde_json::from_slice(RELEASED).unwrap()
}

fn parse(v: &Value) -> Result<QwenConfig, String> {
    QwenConfig::from_json_bytes(&serde_json::to_vec(v).unwrap(), "test").map_err(|e| e.to_string())
}

#[test]
fn the_files_are_the_released_ones() {
    for (file, recorded) in [
        (RELEASED, RECORDED),
        (QWEN35_2B, QWEN35_2B_RECORDED),
        (QWEN35_08B, QWEN35_08B_RECORDED),
    ] {
        let recorded: Value = serde_json::from_str(recorded).unwrap();
        assert_eq!(
            hex::encode(Sha256::digest(file)),
            recorded["files"]["config.json"]["sha256"].as_str().unwrap()
        );
    }
}

#[test]
fn released_config() {
    let c = QwenConfig::from_json_bytes(RELEASED, "config.json").unwrap();
    assert_eq!(
        (c.vocab_size, c.hidden_size, c.intermediate_size),
        (248320, 5120, 17408)
    );
    assert_eq!(
        (c.weights, c.tie_word_embeddings),
        (WeightFormat::Fp8Block128, false)
    );
    assert_eq!(c.rms_norm_eps, 1e-6f64 as f32);
    assert_eq!(c.layers.len(), 64);
    assert_eq!(c.checkpoint_layers, 64);
    for (i, l) in c.layers.iter().enumerate() {
        assert_eq!(l.checkpoint_index, i);
        let want = if i % 4 == 3 {
            LayerKind::Attention
        } else {
            LayerKind::DeltaNet
        };
        assert_eq!(l.kind, want, "layer {i}");
    }
    assert_eq!(c.layers_of(LayerKind::DeltaNet).count(), 48);
    assert_eq!(c.layers_of(LayerKind::Attention).count(), 16);

    let a = &c.attention;
    assert_eq!(
        (a.num_q_heads, a.num_kv_heads, a.head_dim, a.rotary_dim),
        (24, 4, 256, 64)
    );
    assert_eq!(a.group(), 6);
    assert_eq!(a.rope_theta, 1e7);
    assert_eq!(a.softmax_scale, 0.0625);

    let d = &c.deltanet;
    assert_eq!(
        (
            d.num_k_heads,
            d.num_v_heads,
            d.k_head_dim,
            d.v_head_dim,
            d.conv_kernel
        ),
        (16, 48, 128, 128, 4)
    );
    assert_eq!(d.group(), 3);
    assert_eq!(d.conv_dim(), 10240);
    assert_eq!(d.value_dim(), 6144);
    assert_eq!(d.state_len() * 4 * 48, 150_994_944);

    assert_eq!(c.mtp_layers, 1);
    assert_eq!(c.vision_token_ids, [248056, 248057]);
    assert!(c.has_vision);
    assert_eq!(
        (c.bos_token_id, c.eos_token_id),
        (Some(248044), Some(248044))
    );
}

/// Qwen3.5-2B and -0.8B: Qwen3.8-27B's layer pattern and head geometry at 24
/// layers, BF16 weights, a tied head.
#[test]
fn released_qwen35_configs() {
    for (file, hidden, intermediate) in [(QWEN35_2B, 2048, 6144), (QWEN35_08B, 1024, 3584)] {
        let c = QwenConfig::from_json_bytes(file, "config.json").unwrap();
        assert_eq!(
            (c.weights, c.tie_word_embeddings),
            (WeightFormat::Bf16, true)
        );
        assert_eq!(
            (c.vocab_size, c.hidden_size, c.intermediate_size),
            (248320, hidden, intermediate)
        );
        assert_eq!(c.layers.len(), 24);
        for (i, l) in c.layers.iter().enumerate() {
            let want = if i % 4 == 3 {
                LayerKind::Attention
            } else {
                LayerKind::DeltaNet
            };
            assert_eq!(l.kind, want, "layer {i}");
        }
        let a = &c.attention;
        assert_eq!(
            (a.num_q_heads, a.num_kv_heads, a.head_dim, a.rotary_dim),
            (8, 2, 256, 64)
        );
        assert_eq!(a.group(), 4);
        assert_eq!(a.rope_theta, 1e7);
        let d = &c.deltanet;
        assert_eq!(
            (d.num_k_heads, d.num_v_heads, d.k_head_dim, d.v_head_dim),
            (16, 16, 128, 128)
        );
        assert_eq!(d.group(), 1);
        assert_eq!(c.mtp_layers, 1);
        assert_eq!(c.max_position_embeddings, 262_144);
        assert_eq!(c.vision_token_ids, [248056, 248057]);
        assert_eq!((c.bos_token_id, c.eos_token_id), (None, Some(248044)));
    }
}

/// Each change to a released Qwen3.5 config is refused; the keys it lacks
/// and Qwen3.8's carry are accepted with Qwen3.8's values.
#[test]
fn qwen35_configs_are_read_precisely() {
    let base: Value = serde_json::from_slice(QWEN35_2B).unwrap();
    type Edit = fn(&mut Value);
    let refused: Vec<(&str, Edit)> = vec![
        ("a dense MLP list naming a layer", |v| {
            v["text_config"]["mlp_only_layers"] = json!([0])
        }),
        ("the head tied at one level only", |v| {
            v["tie_word_embeddings"] = json!(false)
        }),
        ("the other level", |v| {
            v["text_config"]["tie_word_embeddings"] = json!(false)
        }),
        ("another gate type", |v| {
            v["text_config"]["output_gate_type"] = json!("sigmoid")
        }),
        ("a rotary factor disagreeing", |v| {
            v["text_config"]["partial_rotary_factor"] = json!(0.5)
        }),
        ("unknown text key", |v| {
            v["text_config"]["moe_intermediate_size"] = json!(512)
        }),
        ("FP4 quantisation", |v| {
            v["quantization_config"] = json!({
                "quant_method": "fp4", "fmt": "e2m1", "activation_scheme": "dynamic",
                "weight_block_size": [128, 128], "modules_to_not_convert": []
            })
        }),
    ];
    for (what, edit) in refused {
        let mut v = base.clone();
        edit(&mut v);
        assert!(parse(&v).is_err(), "{what} was accepted");
    }
    type Accepted = fn(&mut Value);
    let accepted: Vec<(&str, Accepted)> = vec![
        ("Qwen3.8's gate key", |v| {
            v["text_config"]["output_gate_type"] = json!("swish")
        }),
        ("Qwen3.8's top-level rotary factor", |v| {
            v["text_config"]["partial_rotary_factor"] = json!(0.25)
        }),
        ("no dense MLP list", |v| {
            v["text_config"]
                .as_object_mut()
                .unwrap()
                .remove("mlp_only_layers");
        }),
        ("an untied head", |v| {
            v["tie_word_embeddings"] = json!(false);
            v["text_config"]["tie_word_embeddings"] = json!(false);
        }),
    ];
    for (what, edit) in accepted {
        let mut v = base.clone();
        edit(&mut v);
        assert!(parse(&v).is_ok(), "{what} was refused");
    }
    let mut untied = base.clone();
    untied["tie_word_embeddings"] = json!(false);
    untied["text_config"]["tie_word_embeddings"] = json!(false);
    assert!(!parse(&untied).unwrap().tie_word_embeddings);
}

#[test]
fn truncation_keeps_any_layers_and_refuses_nonsense() {
    let c = QwenConfig::from_json_bytes(RELEASED, "config.json").unwrap();
    let t = c.clone().truncated(&[0, 1, 2, 3]).unwrap();
    let kinds: Vec<LayerKind> = t.layers.iter().map(|l| l.kind).collect();
    assert_eq!(
        kinds,
        [
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::Attention
        ]
    );
    assert_eq!(t.checkpoint_layers, 64);
    let t = c.clone().truncated(&[63, 5]).unwrap();
    assert_eq!(t.layers[0].checkpoint_index, 63);
    assert!(c.clone().truncated(&[]).is_err());
    assert!(c.clone().truncated(&[64]).is_err());
    assert!(c.clone().truncated(&[2, 2]).is_err());
    assert_eq!(c.without_mtp().mtp_layers, 0);
}

/// Each change to the released config is refused, naming what it refuses.
#[test]
fn unsupported_and_inconsistent_configs_are_refused() {
    type Edit = fn(&mut Value);
    let cases: Vec<(&str, Edit)> = vec![
        ("unknown top-level key", |v| v["rope_scaling"] = json!(null)),
        ("unknown text key", |v| {
            v["text_config"]["sliding_window"] = json!(128)
        }),
        ("unknown rope key", |v| {
            v["text_config"]["rope_parameters"]["factor"] = json!(4)
        }),
        ("unknown quantisation key", |v| {
            v["quantization_config"]["scale_fmt"] = json!("ue8m0")
        }),
        ("unknown vision key", |v| {
            v["vision_config"]["window_size"] = json!(8)
        }),
        ("embeddings tied in the text model only", |v| {
            v["text_config"]["tie_word_embeddings"] = json!(true)
        }),
        ("no output gate", |v| {
            v["text_config"]["attn_output_gate"] = json!(false)
        }),
        ("another gate type", |v| {
            v["text_config"]["output_gate_type"] = json!("sigmoid")
        }),
        ("another rope type", |v| {
            v["text_config"]["rope_parameters"]["rope_type"] = json!("yarn")
        }),
        ("non-interleaved M-RoPE", |v| {
            v["text_config"]["rope_parameters"]["mrope_interleaved"] = json!(false)
        }),
        ("rotary factors disagreeing", |v| {
            v["text_config"]["partial_rotary_factor"] = json!(0.5)
        }),
        ("sections not tiling the frequencies", |v| {
            v["text_config"]["rope_parameters"]["mrope_section"] = json!([11, 11, 11])
        }),
        ("rope theta overflowing f32", |v| {
            v["text_config"]["rope_parameters"]["rope_theta"] = json!(1e40)
        }),
        ("negative eps", |v| {
            v["text_config"]["rms_norm_eps"] = json!(-1e-6)
        }),
        ("layer types against the interval", |v| {
            v["text_config"]["layer_types"][0] = json!("full_attention")
        }),
        ("unknown layer type", |v| {
            v["text_config"]["layer_types"][0] = json!("sliding_attention")
        }),
        ("layer count against the types", |v| {
            v["text_config"]["num_hidden_layers"] = json!(60)
        }),
        ("a bf16 recurrent state", |v| {
            v["text_config"]["mamba_ssm_dtype"] = json!("bfloat16")
        }),
        ("another block size", |v| {
            v["quantization_config"]["weight_block_size"] = json!([64, 64])
        }),
        ("static activations", |v| {
            v["quantization_config"]["activation_scheme"] = json!("static")
        }),
        ("dedicated MTP embeddings", |v| {
            v["text_config"]["mtp_use_dedicated_embeddings"] = json!(true)
        }),
        ("a hidden size off the blocks", |v| {
            v["text_config"]["hidden_size"] = json!(5000)
        }),
        ("value heads not a multiple of key heads", |v| {
            v["text_config"]["linear_num_value_heads"] = json!(40)
        }),
        ("another activation", |v| {
            v["text_config"]["hidden_act"] = json!("gelu")
        }),
        ("another model type", |v| {
            v["model_type"] = json!("qwen3_next")
        }),
    ];
    for (what, edit) in cases {
        let mut v = released();
        edit(&mut v);
        assert!(parse(&v).is_err(), "{what} was accepted");
    }
    assert!(parse(&released()).is_ok());
}
