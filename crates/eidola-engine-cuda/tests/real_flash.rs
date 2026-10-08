//! The CUDA executor on a truncated real MiMo-V2.6-Flash-MOPD checkpoint
//! (checkpoint layers 0, 1, 2 and 5: global + dense FFN, two sliding MoE layers,
//! global MoE) against the f32 reference forward on the same truncation:
//! residual stream after every layer, and logits at every position, for one
//! unchunked prefill, a chunked prefill, and decode steps.
//!
//! Needs a GPU, `EIDOLA_ENGINE_KERNELS_DIR`, `EIDOLA_MIMO_DIR` (a checkpoint
//! directory: the truncated one, or the whole checkpoint) and
//! `EIDOLA_MIMO_GOLDEN` (the model crate's `hf-golden.safetensors`, for its real
//! token ids). Prints the measured differences; run with `--nocapture`.

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use common::setup;
use eidola_engine::executor::{Executor, Maintenance, SeqEntry, StepInput, TableUpdate};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_cuda::{CudaExecutor, CudaExecutorConfig, Gpu, ImageArch};
use eidola_engine_model::compare::compare_logits;
use eidola_engine_model::safetensors::WeightSet;
use eidola_engine_model::{ForwardOptions, LoadOptions, LogitsAt, ModelWeights, ReferenceModel};

const KEEP: [usize; 4] = [0, 1, 2, 5];

/// The checkpoint layers to run: the truncation by default, every layer with
/// `EIDOLA_MIMO_LAYERS=all`.
fn layers() -> Option<Vec<usize>> {
    match std::env::var("EIDOLA_MIMO_LAYERS").as_deref() {
        Ok("all") => None,
        _ => Some(KEEP.to_vec()),
    }
}

/// The checkpoint index of model layer `l`.
fn source(l: usize) -> usize {
    layers().map_or(l, |k| k[l])
}
const BS: u32 = 16;
const SAMPLEABLE: u32 = 151_675;

struct Fixture {
    dir: PathBuf,
    tokens: Vec<u32>,
}

fn fixture() -> Option<Fixture> {
    let (Some(dir), Some(golden)) = (
        std::env::var_os("EIDOLA_MIMO_DIR"),
        std::env::var_os("EIDOLA_MIMO_GOLDEN"),
    ) else {
        eprintln!("skipping: EIDOLA_MIMO_DIR / EIDOLA_MIMO_GOLDEN not set");
        return None;
    };
    let g = WeightSet::open_files(&[PathBuf::from(golden)]).unwrap();
    let tokens = g
        .get("tokens")
        .unwrap()
        .to_i64("tokens")
        .unwrap()
        .into_iter()
        .map(|x| x as u32)
        .collect();
    Some(Fixture {
        dir: dir.into(),
        tokens,
    })
}

fn executor(
    gpu: Gpu,
    dir: &eidola_engine_cuda::KernelDir,
    store: Arc<WeightSet>,
    arch: ImageArch,
) -> CudaExecutor {
    let cfg = CudaExecutorConfig {
        block_size: BS,
        num_blocks: vec![128, 128],
        num_state_slots: 4,
        max_model_len: 1024,
        buckets: vec![Bucket {
            max_seqs: 4,
            max_tokens: 512,
        }],
        sampleable_vocab_size: SAMPLEABLE,
        image: Some(arch),
    };
    CudaExecutor::new(gpu, dir, store, layers().as_deref(), cfg).unwrap()
}

/// Map `slot`'s logical blocks `0..n` to `base + i` in both groups.
fn tables(slot: u32, n: u32, base: u32) -> Vec<TableUpdate> {
    (0..2)
        .flat_map(|group| {
            (0..n).map(move |i| TableUpdate {
                slot,
                group,
                index: i,
                block: base + i,
            })
        })
        .collect()
}

fn row(slot: u32, start: u32, context: u32, n: u32, sample: bool) -> SeqEntry {
    SeqEntry {
        slot,
        token_start: start,
        num_tokens: n,
        context_len: context,
        num_drafts: 0,
        sample,
        sampling: SamplingParams::greedy(),
    }
}

fn step(
    seqs: Vec<SeqEntry>,
    tokens: &[u32],
    m: Vec<Maintenance>,
    t: Vec<TableUpdate>,
) -> StepInput {
    let positions = seqs
        .iter()
        .flat_map(|s| s.context_len..s.context_len + s.num_tokens)
        .collect();
    StepInput {
        bucket: Bucket {
            max_seqs: 4,
            max_tokens: 512,
        },
        maintenance: m,
        table_updates: t,
        seqs,
        token_ids: tokens.to_vec(),
        positions,
        return_logits: false,
    }
}

/// `‖a − b‖ / ‖b‖` and `max|a − b| / max|b|`.
fn rel(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut dmax, mut bmax) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y) as f64;
        num += d * d;
        den += (y as f64) * (y as f64);
        dmax = dmax.max(d.abs());
        bmax = bmax.max((y as f64).abs());
    }
    ((num / den).sqrt(), dmax / bmax)
}

#[test]
fn truncated_flash_matches_the_reference() {
    let Some(fx) = fixture() else { return };
    let Some(su) = setup() else { return };
    let n = 256usize.min(fx.tokens.len());
    let tokens = &fx.tokens[..n];
    let vocab = 152_576usize;

    // Reference.
    let t0 = Instant::now();
    let store = Arc::new(WeightSet::open_dir(&fx.dir).unwrap());
    let mut config = store.model_config().unwrap();
    if let Some(keep) = layers() {
        config = config.truncated(&keep).unwrap();
    }
    let opts = LoadOptions {
        load_mtp: false,
        ..LoadOptions::default()
    };
    let reference = ReferenceModel::new(ModelWeights::load(store.clone(), config, &opts).unwrap());
    let want = reference
        .forward(
            tokens,
            &ForwardOptions {
                logits: LogitsAt::All,
                capture_layers: true,
            },
        )
        .unwrap();
    println!("reference forward over {n} tokens: {:.1?}", t0.elapsed());
    let t0 = Instant::now();
    let (q_layers, q_logits) = common::qref::forward(&reference, tokens);
    println!(
        "quantization-emulating reference forward: {:.1?}",
        t0.elapsed()
    );
    println!("quantization emulated vs f32 reference: layer | rel L2 | max|Δ|/max|ref|");
    for (l, got) in q_layers.iter().enumerate() {
        let (l2, mx) = rel(&got.data, &want.layer_outputs[l].data);
        println!("  {l} (checkpoint {}) | {l2:.3e} | {mx:.3e}", source(l));
    }
    let q_agree = compare_logits(&want.logits.data, &q_logits.data, vocab).unwrap();
    println!("quantization emulated vs f32 reference, logits: {q_agree}");
    drop(reference);

    let gpu = su.gpu;
    // `EIDOLA_IMAGES=exact` runs only the device's own image (the full model
    // takes minutes to load).
    let mut archs = su.archs.clone();
    if std::env::var("EIDOLA_IMAGES").as_deref() == Ok("exact") {
        archs.truncate(1);
    }
    let dir = su.dir;
    let mut gpu_slot = Some(gpu);
    for arch in archs {
        let gpu = match gpu_slot.take() {
            Some(g) => g,
            None => Gpu::open(0).unwrap(),
        };
        let t0 = Instant::now();
        let mut ex = executor(gpu, &dir, store.clone(), arch);
        println!("{arch:?}: executor loaded in {:.1?}", t0.elapsed());
        ex.model_mut().capture_layers = true;
        ex.record_all_logits = true;

        // Unchunked prefill in slot 0 (blocks 1..).
        let nb = (n as u32).div_ceil(BS);
        let t0 = Instant::now();
        let out = ex
            .execute(&step(
                vec![row(0, 0, 0, n as u32, true)],
                tokens,
                vec![],
                tables(0, nb, 1),
            ))
            .unwrap();
        println!("{arch:?}: prefill of {n} tokens in {:.1?}", t0.elapsed());
        let logits = ex.take_all_logits().unwrap();
        println!("{arch:?}: layer | rel L2 | max|Δ|/max|ref|");
        for (l, got) in ex.model().captured.iter().enumerate() {
            let (l2, mx) = rel(got, &want.layer_outputs[l].data);
            println!(
                "{arch:?}:   {l} (checkpoint {}) | {l2:.3e} | {mx:.3e}",
                source(l)
            );
        }
        println!("{arch:?}: vs quantization emulation: layer | rel L2 | max|Δ|/max|ref|");
        for (l, got) in ex.model().captured.iter().enumerate() {
            let (l2, mx) = rel(got, &q_layers[l].data);
            println!(
                "{arch:?}:   {l} (checkpoint {}) | {l2:.3e} | {mx:.3e}",
                source(l)
            );
        }
        let agree = compare_logits(&want.logits.data, &logits, vocab).unwrap();
        println!("{arch:?}: logits vs reference (unchunked): {agree}");
        let agree_q = compare_logits(&q_logits.data, &logits, vocab).unwrap();
        println!("{arch:?}: logits vs quantization emulation (unchunked): {agree_q}");
        assert!(agree.top1_rate() > 0.9, "{agree}");
        assert!(out.row(0)[0] < SAMPLEABLE);

        // The same tokens chunked (37, 64, then the rest) in slot 1, which
        // starts from fresh blocks.
        let base = 1 + nb;
        let mut chunked = Vec::new();
        let mut c = 0u32;
        let mut updates = tables(1, nb, base);
        for len in [37u32, 64, n as u32 - 101] {
            let toks = &tokens[c as usize..(c + len) as usize];
            ex.execute(&step(
                vec![row(1, 0, c, len, c + len == n as u32)],
                toks,
                vec![],
                std::mem::take(&mut updates),
            ))
            .unwrap();
            chunked.extend(ex.take_all_logits().unwrap());
            c += len;
        }
        let agree_c = compare_logits(&want.logits.data, &chunked, vocab).unwrap();
        let self_c = compare_logits(&logits, &chunked, vocab).unwrap();
        println!("{arch:?}: logits vs reference (chunked 37/64/rest): {agree_c}");
        println!("{arch:?}: chunked vs unchunked on the GPU: {self_c}");

        // Decode: prefill n - 8 tokens in slot 2, then 8 single-token steps.
        let base = base + nb;
        let pre = n as u32 - 8;
        ex.execute(&step(
            vec![row(2, 0, 0, pre, false)],
            &tokens[..pre as usize],
            vec![],
            tables(2, nb, base),
        ))
        .unwrap();
        let mut decoded = Vec::new();
        for p in pre..n as u32 {
            ex.execute(&step(
                vec![row(2, 0, p, 1, true)],
                &tokens[p as usize..p as usize + 1],
                vec![],
                vec![],
            ))
            .unwrap();
            decoded.extend(ex.take_all_logits().unwrap());
        }
        let want_dec = &want.logits.data[pre as usize * vocab..];
        let agree_d = compare_logits(want_dec, &decoded, vocab).unwrap();
        println!("{arch:?}: logits vs reference (8 decode steps): {agree_d}");

        ex.model_mut().capture_layers = false;
        gpu_slot = None;
        drop(ex);
    }
}

/// The emulation's e4m3 rounding reproduces every finite e4m3 value exactly
/// and rounds halfway cases to even.
#[test]
fn e4m3_rounding_is_exact_on_codes() {
    use eidola_engine_model::numeric::fp8_e4m3_to_f32;
    for code in 0u8..=255 {
        if code & 0x7f == 0x7f {
            continue;
        }
        let v = fp8_e4m3_to_f32(code);
        assert_eq!(common::qref::e4m3(v), v, "code {code:#04x}");
    }
    assert_eq!(common::qref::e4m3(1000.0), 448.0);
    // Between 1 (code 0x38) and 1.125: the midpoint goes to the even mantissa.
    assert_eq!(common::qref::e4m3(1.0625), 1.0);
    assert_eq!(common::qref::e4m3(1.1875), 1.25);
}
