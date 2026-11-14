//! The DeepSeek-V4.1 reference forward against the committed synthetic
//! fixture, whose goldens are the released model code run over an emulated
//! kernel layer with its library arithmetic pinned (see `golden/README.md`).
//! Under those rules the reference is a function of the tokens alone, and
//! every comparison here is exact.

use std::path::PathBuf;

use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::forward::{DsForwardOptions, LayerOptions};
use eidola_engine_model::deepseek::{DsLoadOptions, DsModel, load_reference};
use eidola_engine_model::safetensors::Dtype;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/deepseek_tiny")
}

struct Golden(WeightSet);

impl Golden {
    fn open() -> Self {
        Golden(WeightSet::open_files(&[fixture().join("golden.safetensors")]).unwrap())
    }
    fn f32(&self, name: &str) -> Vec<f32> {
        self.0.get(name).unwrap().to_f32(name).unwrap()
    }
    fn ints(&self, name: &str) -> Vec<i64> {
        self.0.get(name).unwrap().to_i64(name).unwrap()
    }
    fn tokens(&self) -> Vec<u32> {
        self.ints("tokens").into_iter().map(|t| t as u32).collect()
    }
    fn token_map(&self) -> Vec<u32> {
        self.ints("engram.token_map")
            .into_iter()
            .map(|t| t as u32)
            .collect()
    }
}

fn model(golden: &Golden) -> DsModel {
    load_reference(
        &fixture(),
        Ok,
        &DsLoadOptions::default(),
        Some(golden.token_map()),
    )
    .unwrap()
}

/// Exact equality (`-0.0 == 0.0`), reporting the first differing row.
#[track_caller]
fn assert_same(what: &str, got: &[f32], want: &[f32], row_width: usize) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        let differing = got.iter().zip(want).filter(|(a, b)| a != b).count();
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

fn run_stepwise(model: &DsModel, tokens: &[u32]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let layers = model.config().layers.len();
    let mut state = model.new_state();
    let mut taps = vec![Vec::new(); layers];
    let (mut logits, mut normed, mut hidden) = (Vec::new(), Vec::new(), Vec::new());
    for t in tokens {
        let out = model
            .forward(
                &mut state,
                &[*t],
                &DsForwardOptions {
                    taps: true,
                    ..Default::default()
                },
            )
            .unwrap();
        for (l, tap) in out.taps.iter().enumerate() {
            taps[l].extend_from_slice(&tap.data);
        }
        logits.extend_from_slice(&out.logits.data);
        normed.extend_from_slice(&out.normed.data);
        hidden.extend_from_slice(&out.main_hidden.unwrap().data);
    }
    (taps, logits, normed, hidden)
}

#[test]
fn engram_rows_match_the_reference_hash() {
    let golden = Golden::open();
    let model = model(&golden);
    let tokens = golden.tokens();
    let rows = model.embed(&mut model.new_state(), &tokens).unwrap();
    let want = golden.ints("step.engram_rows");
    let lookups = model.config().engram.as_ref().unwrap().lookups();
    for (slot, got) in rows.engram_rows.iter().enumerate() {
        for (p, got) in got.chunks(lookups).enumerate() {
            let at = (p * 2 + slot) * lookups;
            let want: Vec<u64> = want[at..at + lookups].iter().map(|r| *r as u64).collect();
            assert_eq!(got, want, "position {p} Engram layer {slot}");
        }
    }
}

/// One token per call, the reference's decode pattern: every layer's
/// residual streams, the final normed state, the drafter's input and the
/// logits equal the reference exactly.
#[test]
fn stepwise_forward_matches_the_reference_exactly() {
    let golden = Golden::open();
    let model = model(&golden);
    let c = model.config();
    let (taps, logits, normed, hidden) = run_stepwise(&model, &golden.tokens());
    for (l, tap) in taps.iter().enumerate() {
        assert_same(
            &format!("streams after layer {l}"),
            tap,
            &golden.f32(&format!("step.layer.{l}")),
            c.hc_mult * c.dim,
        );
    }
    assert_same("normed", &normed, &golden.f32("step.normed"), c.dim);
    assert_same(
        "main_hidden",
        &hidden,
        &golden.f32("step.main_hidden"),
        hidden.len() / golden.tokens().len(),
    );
    assert_same("logits", &logits, &golden.f32("step.logits"), c.vocab_size);
}

/// The sub-layer outputs, layer by layer: where a regression first shows.
#[test]
fn sub_layer_outputs_match_the_reference_exactly() {
    let golden = Golden::open();
    let model = model(&golden);
    let c = model.config();
    let mut state = model.new_state();
    let mut rows = model.embed(&mut state, &golden.tokens()).unwrap();
    for l in 0..c.layers.len() {
        let out = model
            .layer_forward(l, &mut state, &rows, LayerOptions::default())
            .unwrap();
        assert_same(
            &format!("attention output of layer {l}"),
            &out.attention.data,
            &golden.f32(&format!("step.attn.{l}")),
            c.dim,
        );
        assert_same(
            &format!("feed-forward output of layer {l}"),
            &out.ffn.data,
            &golden.f32(&format!("step.ffn.{l}")),
            c.dim,
        );
        rows = out.rows;
    }
}

/// The whole sequence as one chunk equals one token per call, and equals the
/// reference run as one prefill chunk: at this fixture's window (48, under one
/// 64-slot block) the reference's prefill and decode layouts put the same
/// entries in the same blocks whenever the chunk is at least a window long.
/// At the released window of 128 they do not (`tests/deepseek_real.rs`).
#[test]
fn one_chunk_equals_stepwise_and_the_reference_prefill() {
    let golden = Golden::open();
    let model = model(&golden);
    let c = model.config();
    let tokens = golden.tokens();
    let (taps, logits, ..) = run_stepwise(&model, &tokens);
    let out = model
        .forward(
            &mut model.new_state(),
            &tokens,
            &DsForwardOptions {
                taps: true,
                ..Default::default()
            },
        )
        .unwrap();
    for (l, tap) in out.taps.iter().enumerate() {
        assert_same(
            &format!("chunked layer {l}"),
            &tap.data,
            &taps[l],
            c.hc_mult * c.dim,
        );
        assert_same(
            &format!("prefill layer {l}"),
            &tap.data,
            &golden.f32(&format!("prefill.layer.{l}")),
            c.hc_mult * c.dim,
        );
    }
    assert_same("chunked logits", &out.logits.data, &logits, c.vocab_size);
    assert_same(
        "prefill logits",
        &out.logits.data,
        &golden.f32("prefill.logits"),
        c.vocab_size,
    );
    // Uneven chunks, one of them shorter than the compressor's group.
    let mut state = model.new_state();
    let mut chunked = Vec::new();
    for chunk in [&tokens[..1], &tokens[1..38], &tokens[38..39], &tokens[39..]] {
        chunked.extend(
            model
                .forward(&mut state, chunk, &Default::default())
                .unwrap()
                .logits
                .data,
        );
    }
    assert_same("uneven chunks", &chunked, &logits, c.vocab_size);
}

#[test]
fn drafter_matches_the_reference_exactly() {
    let golden = Golden::open();
    let model = model(&golden);
    let c = model.config();
    let tokens = golden.tokens();
    let n = tokens.len();
    let hidden = golden.f32("step.main_hidden");
    let width = hidden.len() / n;
    let slots = c.drafter.as_ref().unwrap().slots;
    let (want_tokens, want_logits, want_conf) = (
        golden.ints("step.draft_tokens"),
        golden.f32("step.draft_logits"),
        golden.f32("step.draft_confidence"),
    );
    let mut state = model.new_drafter_state();
    model.drafter_observe(&mut state, &hidden[..width]).unwrap();
    for p in 1..n {
        let given = tokens[(p + 1) % n];
        let core = model
            .drafter_forward(&mut state, &hidden[p * width..(p + 1) * width], given)
            .unwrap();
        let draft = model.draft_greedy(&core, given);
        let at = (p - 1) * slots;
        let want: Vec<u32> = want_tokens[at..at + slots]
            .iter()
            .map(|t| *t as u32)
            .collect();
        assert_eq!(draft.tokens, want, "draft tokens at position {p}");
        assert_same(
            &format!("draft logits at position {p}"),
            &draft.logits.data,
            &want_logits[at * c.vocab_size..(at + slots) * c.vocab_size],
            c.vocab_size,
        );
        assert_same(
            &format!("draft confidence at position {p}"),
            &draft.confidence,
            &want_conf[at..at + slots],
            1,
        );
    }
}

/// The reference lays a prefill chunk shorter than the window out
/// differently: the global entries follow the chunk's positions directly, so
/// they fall into other 64-wide blocks than in the canonical layout. With
/// that layout reproduced the forward equals the reference's short prefill
/// exactly; the canonical layout does not, and the difference is only that.
#[test]
fn prefill_layout_is_the_only_difference_from_a_short_reference_prefill() {
    let golden = Golden::open();
    let model = model(&golden);
    let v = model.config().vocab_size;
    let want = golden.f32("short.logits");
    let chunk = want.len() / v;
    assert!(chunk < model.config().attention.window);
    let tokens = &golden.tokens()[..chunk];
    let run = |reference_prefill| {
        let opts = DsForwardOptions {
            reference_prefill,
            ..Default::default()
        };
        model
            .forward(&mut model.new_state(), tokens, &opts)
            .unwrap()
            .logits
            .data
    };
    assert_same("short-prefill layout", &run(Some(chunk)), &want, v);
    let canonical = run(None);
    let differing = canonical
        .chunks(v)
        .zip(want.chunks(v))
        .filter(|(a, b)| a != b)
        .count();
    let a = eidola_engine_model::compare::compare_logits(&want, &canonical, v).unwrap();
    println!(
        "canonical layout against a {chunk}-token reference prefill: {differing} of {chunk} positions differ; {a}"
    );
    // More than 16 global entries are needed before the layouts part.
    assert!(differing > 0 && canonical[..16 * v] == want[..16 * v]);
}

/// The two halves run separately, with only the boundary value between them,
/// equal the single full forward.
#[test]
fn split_halves_equal_the_full_forward() {
    let golden = Golden::open();
    let model = model(&golden);
    let c = model.config();
    let tokens = golden.tokens();
    let opts = DsForwardOptions {
        taps: true,
        ..Default::default()
    };
    let full = model
        .forward(&mut model.new_state(), &tokens, &opts)
        .unwrap();
    // Layer 4 is the ratio-1 KV source the later layers read, as layer 20 is
    // in the released model.
    let split = 4;
    assert!(c.layers[split].is_kv_source && c.layers[split].is_candidate_source);
    for chunks in [vec![tokens.len()], vec![1, 30, 1, 64]] {
        let (mut encoder, mut decoder) = (model.new_state(), model.new_state());
        let (mut logits, mut last_tap, mut at) = (Vec::new(), Vec::new(), 0);
        for len in chunks {
            let boundary = model
                .encode(&mut encoder, &tokens[at..at + len], split, &opts)
                .unwrap();
            assert_eq!(boundary.bank.as_ref().unwrap().entries, at + len);
            let out = model.decode(&mut decoder, &boundary, split, &opts).unwrap();
            logits.extend(out.logits.data);
            last_tap.extend(out.taps.last().unwrap().data.clone());
            at += len;
        }
        assert_same("split logits", &logits, &full.logits.data, c.vocab_size);
        assert_same(
            "split last layer",
            &last_tap,
            &full.taps.last().unwrap().data,
            c.hc_mult * c.dim,
        );
    }
}

/// A triangular replay: the last position's logits from the decoder layers
/// run only where the window chain needs them. Decoder layer `j` (of `d`)
/// emits positions `t - (d-1-j)·(w-1) ..= t` and takes the `w - 1` rows before
/// those for their window KV only. The result equals the full forward bit for
/// bit.
#[test]
fn triangular_decoder_replay_equals_the_full_forward() {
    let golden = Golden::open();
    let tokens = golden.tokens();
    // A narrower window than the fixture's, so the triangle is narrower than
    // the sequence: the same weights are a valid model at any window.
    let config = std::fs::read_to_string(fixture().join("config.json"))
        .unwrap()
        .replace("\"sliding_window\": 48", "\"sliding_window\": 12");
    let dir = std::env::temp_dir().join(format!("eidola-deepseek-triangle-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config).unwrap();
    std::fs::copy(
        fixture().join("model.safetensors"),
        dir.join("model.safetensors"),
    )
    .unwrap();
    let model = load_reference(
        &dir,
        Ok,
        &DsLoadOptions::default(),
        Some(golden.token_map()),
    )
    .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let c = model.config();
    let (w, layers, split) = (c.attention.window, c.layers.len(), 4);
    let full = model
        .forward(&mut model.new_state(), &tokens, &Default::default())
        .unwrap();

    let boundary = model
        .encode(&mut model.new_state(), &tokens, split, &Default::default())
        .unwrap();
    let t = tokens.len() - 1;
    let depth = layers - split;
    assert!(
        depth * (w - 1) < t,
        "the triangle must be narrower than the sequence"
    );
    let mut state = model.new_state();
    state.supply_bank(split, boundary.bank.clone().unwrap());
    let mut rows = boundary.rows.from_position(t - depth * (w - 1));
    for (j, layer) in (split..layers).enumerate() {
        let emit_from = t - (depth - 1 - j) * (w - 1);
        assert_eq!(rows.start, emit_from - (w - 1));
        rows = model
            .layer_forward(
                layer,
                &mut state,
                &rows,
                LayerOptions {
                    emit_from,
                    ..Default::default()
                },
            )
            .unwrap()
            .rows;
        assert_eq!((rows.start, rows.len()), (emit_from, t - emit_from + 1));
    }
    assert_eq!(rows.len(), 1);
    let (_, logits) = model.head(&rows, true);
    assert_same(
        "triangular logits",
        &logits.data,
        full.logits.row(t),
        c.vocab_size,
    );

    // One row too few below a layer is refused, not answered with a shorter window.
    let mut state = model.new_state();
    state.supply_bank(split, boundary.bank.clone().unwrap());
    let short = boundary.rows.from_position(t - (w - 2));
    let err = model.layer_forward(
        split,
        &mut state,
        &short,
        LayerOptions {
            emit_from: t,
            ..Default::default()
        },
    );
    assert!(err.is_err());
    // With an explicit window start the same rows are a bounded replay.
    let mut state = model.new_state();
    state.supply_bank(split, boundary.bank.clone().unwrap());
    let bounded = model.layer_forward(
        split,
        &mut state,
        &short,
        LayerOptions {
            emit_from: t,
            window_start: t - (w - 2),
            reference_prefill: None,
        },
    );
    assert!(bounded.is_ok());
}

#[test]
fn fixture_tensors_use_the_checkpoint_formats() {
    let store = WeightSet::open_dir(&fixture()).unwrap();
    let dtype = |n: &str| store.get(n).unwrap().dtype;
    assert_eq!(dtype("layers.0.attn.wq_a.weight"), Dtype::F8E4M3);
    assert_eq!(dtype("layers.0.attn.wq_a.scale"), Dtype::F8E8M0);
    assert_eq!(dtype("layers.0.ffn.experts.0.w1.weight"), Dtype::I8);
    assert_eq!(dtype("layers.1.engram.embed.weight"), Dtype::F8E4M3);
    assert_eq!(dtype("layers.0.hc_attn_fn"), Dtype::F32);
    assert_eq!(dtype("embed.weight"), Dtype::Bf16);
}
