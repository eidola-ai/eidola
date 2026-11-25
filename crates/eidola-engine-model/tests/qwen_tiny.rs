//! The Qwen3.8 and Qwen3.5 reference forward against the committed synthetic
//! fixtures, whose goldens are Transformers' `qwen3_5` model code run with its
//! unspecified operations pinned (see `golden/qwen/README.md`). Under those
//! rules the reference is a function of the tokens alone, and every
//! comparison here is exact.
//!
//! Every test runs on each fixture, in a module of its name: `qwen_tiny`
//! (Qwen3.8's FP8 storage), `qwen_tiny_bf16` (Qwen3.5's BF16 storage, its own
//! head) and `qwen_tiny_bf16_tied` (BF16, the head tied to the embedding).

use std::path::PathBuf;

use eidola_engine_model::WeightSet;
use eidola_engine_model::qwen::{
    KvCache, LayerState, LogitRows, QwenForwardOptions, QwenModel, SequenceState, WeightFormat,
    load_reference,
};
use eidola_engine_model::tensor::Matrix;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

struct Golden(WeightSet);

impl Golden {
    fn open(name: &str) -> Self {
        Golden(WeightSet::open_files(&[fixture(name).join("golden.safetensors")]).unwrap())
    }
    fn f32(&self, name: &str) -> Vec<f32> {
        self.0.get(name).unwrap().to_f32(name).unwrap()
    }
    fn tokens(&self) -> Vec<u32> {
        let v = self.0.get("tokens").unwrap().to_i64("tokens").unwrap();
        v.into_iter().map(|t| t as u32).collect()
    }
}

fn model(name: &str) -> QwenModel {
    load_reference(&fixture(name), Ok).unwrap()
}

/// Bit equality, reporting the first differing element.
#[track_caller]
fn assert_bits(what: &str, got: &[f32], want: &[f32], row_width: usize) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got
        .iter()
        .zip(want)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        let differing = got
            .iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        panic!(
            "{what}: {differing} of {} elements differ; first at row {} column {}: got {} want {}",
            got.len(),
            i / row_width,
            i % row_width,
            got[i],
            want[i]
        );
    }
}

fn all(keep_streams: bool) -> QwenForwardOptions {
    QwenForwardOptions {
        logits: LogitRows::All,
        keep_streams,
    }
}

/// The DeltaNet states of a sequence state, by layer.
fn deltanet_states(state: &SequenceState) -> Vec<(usize, &[f32], &[f32])> {
    state
        .layers
        .iter()
        .enumerate()
        .filter_map(|(i, l)| match l {
            LayerState::DeltaNet(s) => Some((i, &s.conv[..], &s.recurrent[..])),
            LayerState::Attention(_) => None,
        })
        .collect()
}

fn one_call_matches_the_reference_exactly(f: &str) {
    let g = Golden::open(f);
    let m = model(f);
    let tokens = g.tokens();
    let mut state = m.new_state();
    let out = m.forward(&mut state, &tokens, &all(true)).unwrap();
    let h = m.config().hidden_size;
    for (i, s) in out.streams.iter().enumerate() {
        assert_bits(
            &format!("layer {i} stream"),
            &s.data,
            &g.f32(&format!("prefill.stream.{i}")),
            h,
        );
    }
    assert_bits("normed", &out.normed.data, &g.f32("prefill.normed"), h);
    assert_bits(
        "logits",
        &out.logits.data,
        &g.f32("prefill.logits"),
        m.config().vocab_size,
    );
    let states = deltanet_states(&state);
    assert_eq!(states.len(), 3);
    for (i, conv, rec) in states {
        let c = m.config().deltanet.conv_dim();
        assert_bits(
            &format!("layer {i} conv state"),
            conv,
            &g.f32(&format!("state.{i}.conv")),
            c,
        );
        assert_bits(
            &format!("layer {i} recurrent state"),
            rec,
            &g.f32(&format!("state.{i}.recurrent")),
            m.config().deltanet.v_head_dim,
        );
    }
}

fn one_token_per_call_matches_the_reference_exactly(f: &str) {
    let g = Golden::open(f);
    let m = model(f);
    let tokens = g.tokens();
    let first = 24;
    let mut state = m.new_state();
    let mut logits = m
        .forward(&mut state, &tokens[..first], &all(false))
        .unwrap()
        .logits
        .data;
    for t in &tokens[first..] {
        logits.extend(
            m.forward(&mut state, &[*t], &all(false))
                .unwrap()
                .logits
                .data,
        );
    }
    assert_bits(
        "stepwise logits",
        &logits,
        &g.f32("step.logits"),
        m.config().vocab_size,
    );
    assert_bits(
        "stepwise against one call",
        &logits,
        &g.f32("prefill.logits"),
        m.config().vocab_size,
    );
}

fn mtp_matches_the_reference_exactly(f: &str) {
    let g = Golden::open(f);
    let m = model(f);
    let tokens = g.tokens();
    let h = m.config().hidden_size;
    let normed = g.f32("prefill.normed");
    let n = tokens.len() - 1;
    let hidden = Matrix::from_vec(n, h, normed[..n * h].to_vec());
    let mut cache = KvCache::default();
    let out = m
        .mtp_forward(0, &mut cache, 0, &tokens[1..], &hidden, &LogitRows::All)
        .unwrap();
    assert_bits("MTP hidden", &out.hidden.data, &g.f32("mtp.hidden"), h);
    assert_bits("MTP normed", &out.normed.data, &g.f32("mtp.normed"), h);
    assert_bits(
        "MTP logits",
        &out.logits.data,
        &g.f32("mtp.logits"),
        m.config().vocab_size,
    );

    // The same rows one at a time, each from the cache the rows before left.
    let mut cache = KvCache::default();
    let mut logits = Vec::new();
    for r in 0..n {
        let row = Matrix::from_vec(1, h, hidden.row(r).to_vec());
        let o = m
            .mtp_forward(
                0,
                &mut cache,
                r,
                &tokens[r + 1..r + 2],
                &row,
                &LogitRows::All,
            )
            .unwrap();
        logits.extend(o.logits.data);
    }
    assert_bits(
        "MTP one row per call",
        &logits,
        &g.f32("mtp.logits"),
        m.config().vocab_size,
    );
}

fn prefix_rows_are_bit_identical(f: &str) {
    let g = Golden::open(f);
    let m = model(f);
    let tokens = g.tokens();
    let v = m.config().vocab_size;
    let full = m
        .forward(&mut m.new_state(), &tokens, &all(false))
        .unwrap()
        .logits;
    for n in [1, 5, 17, 33, 64] {
        let part = m
            .forward(&mut m.new_state(), &tokens[..n], &all(false))
            .unwrap()
            .logits;
        assert_bits(
            &format!("first {n} rows"),
            &part.data,
            &full.data[..n * v],
            v,
        );
    }
}

/// Running 0..n, saving the state, then running n..m from the saved copy
/// equals running 0..m, in logits and in every layer's state, at any n:
/// chunk ends, mid-chunk points and single tokens alike.
fn a_saved_state_resumes_exactly(f: &str) {
    let g = Golden::open(f);
    let m = model(f);
    let tokens = &g.tokens()[..56];
    let v = m.config().vocab_size;
    let mut whole_state = m.new_state();
    let whole = m.forward(&mut whole_state, tokens, &all(false)).unwrap();
    for (n, chunks) in [
        (1usize, vec![55usize]),
        (3, vec![1, 1, 50, 1]),
        (13, vec![7, 36]),
        (29, vec![27]),
        (55, vec![1]),
    ] {
        let mut state = m.new_state();
        let mut logits = m
            .forward(&mut state, &tokens[..n], &all(false))
            .unwrap()
            .logits
            .data;
        let saved = state.clone();
        drop(state);
        let mut resumed = saved.clone();
        let mut at = n;
        for c in chunks {
            logits.extend(
                m.forward(&mut resumed, &tokens[at..at + c], &all(false))
                    .unwrap()
                    .logits
                    .data,
            );
            at += c;
        }
        assert_eq!(at, tokens.len());
        assert_bits(&format!("resume at {n}"), &logits, &whole.logits.data, v);
        assert_eq!(resumed.position, whole_state.position);
        for ((i, conv, rec), (_, wconv, wrec)) in deltanet_states(&resumed)
            .into_iter()
            .zip(deltanet_states(&whole_state))
        {
            assert_bits(&format!("resume at {n}: layer {i} conv"), conv, wconv, 1);
            assert_bits(&format!("resume at {n}: layer {i} state"), rec, wrec, 1);
        }
        assert!(resumed == whole_state, "resume at {n}: KV differs");
    }
}

fn vision_placeholders_and_out_of_range_tokens_are_refused(f: &str) {
    let m = model(f);
    let v = m.config().vocab_size as u32;
    for bad in [v - 2, v - 1, v] {
        assert!(
            m.forward(&mut m.new_state(), &[1, bad], &all(false))
                .is_err(),
            "token {bad}"
        );
    }
}

/// Which format each fixture stores, and that a tied head is the embedding's
/// own rows.
fn fixture_is_its_format(f: &str) {
    let m = model(f);
    let c = m.config();
    let (format, tied) = match f {
        "qwen_tiny" => (WeightFormat::Fp8Block128, false),
        "qwen_tiny_bf16" => (WeightFormat::Bf16, false),
        "qwen_tiny_bf16_tied" => (WeightFormat::Bf16, true),
        other => panic!("fixture {other}"),
    };
    assert_eq!((c.weights, c.tie_word_embeddings), (format, tied));
    let w = &m.weights;
    let same_rows = (0..c.vocab_size).all(|r| w.head.row(r) == w.embed.row(r));
    assert_eq!(same_rows, tied, "{f}: the head is the embedding");
}

macro_rules! on_each_fixture {
    ($($fixture:ident),* ; $tests:tt) => {
        $(on_each_fixture!(@one $fixture $tests);)*
    };
    (@one $fixture:ident [$($test:ident),*]) => {
        mod $fixture {
            $(
                #[test]
                fn $test() {
                    super::$test(stringify!($fixture))
                }
            )*
        }
    };
}

on_each_fixture!(qwen_tiny, qwen_tiny_bf16, qwen_tiny_bf16_tied; [
    one_call_matches_the_reference_exactly,
    one_token_per_call_matches_the_reference_exactly,
    mtp_matches_the_reference_exactly,
    prefix_rows_are_bit_identical,
    a_saved_state_resumes_exactly,
    vision_placeholders_and_out_of_range_tokens_are_refused,
    fixture_is_its_format
]);

/// A tied config over a checkpoint that carries a head of its own is refused,
/// and so is an untied config over a checkpoint without one: the head the
/// forward reads is never a guess.
#[test]
fn a_head_that_disagrees_with_the_config_is_refused() {
    let dir = std::env::temp_dir().join(format!("qwen-tied-mix-{}", std::process::id()));
    for (config, weights) in [
        ("qwen_tiny_bf16_tied", "qwen_tiny_bf16"),
        ("qwen_tiny_bf16", "qwen_tiny_bf16_tied"),
    ] {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(fixture(config).join("config.json"), dir.join("config.json")).unwrap();
        std::fs::copy(
            fixture(weights).join("model.safetensors"),
            dir.join("model.safetensors"),
        )
        .unwrap();
        let err = load_reference(&dir, Ok).err();
        std::fs::remove_dir_all(&dir).unwrap();
        let err = err.unwrap_or_else(|| panic!("{config} over {weights} loaded"));
        assert!(err.to_string().contains("lm_head"), "{err}");
    }
}
