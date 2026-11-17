//! Qwen3.8 configuration: the released `config.json`, its per-layer table,
//! truncation, and refusals.
//!
//! `tests/data/qwen3.8-27b.config.json` is the file of `Qwen/Qwen3.8-27B-FP8`
//! at revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`, byte for byte (its
//! SHA-256 is checked against the recorded one below).

use eidola_engine_model::qwen::QwenConfig;
use eidola_engine_model::qwen::config::LayerKind;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const RELEASED: &[u8] = include_bytes!("data/qwen3.8-27b.config.json");
const RECORDED: &str = include_str!("../golden/qwen/upstream-sha256.json");

fn released() -> Value {
    serde_json::from_slice(RELEASED).unwrap()
}

fn parse(v: &Value) -> Result<QwenConfig, String> {
    QwenConfig::from_json_bytes(&serde_json::to_vec(v).unwrap(), "test").map_err(|e| e.to_string())
}

#[test]
fn the_file_is_the_released_one() {
    let recorded: Value = serde_json::from_str(RECORDED).unwrap();
    assert_eq!(
        hex::encode(Sha256::digest(RELEASED)),
        recorded["files"]["config.json"]["sha256"].as_str().unwrap()
    );
}

#[test]
fn released_config() {
    let c = QwenConfig::from_json_bytes(RELEASED, "config.json").unwrap();
    assert_eq!(
        (c.vocab_size, c.hidden_size, c.intermediate_size),
        (248320, 5120, 17408)
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
        ("tied embeddings", |v| {
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
