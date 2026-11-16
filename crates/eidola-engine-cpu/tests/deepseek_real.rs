//! The DeepSeek-V4.1 CPU executor on a truncation of the released checkpoint with every
//! mechanism at its released width (ignored: needs the fetched weights and goldens, see
//! the model crate's `golden/deepseek/README.md`).
//!
//!     EIDOLA_DEEPSEEK_TRUNCATED_DIR=<dir> cargo test --release -p eidola-engine-cpu \
//!         --test deepseek_real -- --ignored --nocapture

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cpu::deepseek::{DsCpuExecutor, DsCpuExecutorConfig};
use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::engram::{EngramTable, SparseEngramTable};
use eidola_engine_model::deepseek::forward::DsForwardOptions;
use eidola_engine_model::deepseek::{DsLoadOptions, DsModel, DsWeights, config_of};

/// The truncation `golden/deepseek/deepseek_golden/truncated.py` fetches.
const LAYERS: [usize; 12] = [0, 1, 2, 3, 14, 15, 20, 21, 24, 37, 38, 39];
const BACKBONE_EXPERTS: usize = 32;
const DRAFTER_EXPERTS: usize = 16;

fn open(path: &Path) -> WeightSet {
    WeightSet::open_files(&[path.to_path_buf()]).unwrap()
}

fn ints(set: &WeightSet, name: &str) -> Vec<i64> {
    set.get(name).unwrap().to_i64(name).unwrap()
}

/// The truncated model and the goldens' token sequence, whose Engram rows are the ones
/// fetched.
fn load() -> (Arc<DsModel>, Vec<u32>) {
    let dir = PathBuf::from(
        std::env::var("EIDOLA_DEEPSEEK_TRUNCATED_DIR")
            .expect("EIDOLA_DEEPSEEK_TRUNCATED_DIR names the fetched checkpoint"),
    );
    let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
    let pinned = open(&dir.join("golden/reference-pinned.safetensors"));
    let tokens: Vec<u32> = ints(&pinned, "tokens").iter().map(|t| *t as u32).collect();
    let token_map = ints(&pinned, "engram.token_map")
        .iter()
        .map(|t| *t as u32)
        .collect();
    let config = config_of(&store)
        .unwrap()
        .truncated(&LAYERS)
        .unwrap()
        .with_routed_experts(BACKBONE_EXPERTS, DRAFTER_EXPERTS)
        .unwrap();
    let rows = open(&dir.join("golden/engram-rows.safetensors"));
    let spec = config.engram.clone().unwrap();
    let engram_tables = spec
        .layers
        .iter()
        .filter(|l| LAYERS.contains(&l.source_layer))
        .map(|l| {
            let p = format!("layers.{}", l.source_layer);
            let ids: Vec<u64> = ints(&rows, &format!("{p}.rows"))
                .iter()
                .map(|r| *r as u64)
                .collect();
            let table = SparseEngramTable::from_stored(
                l.num_rows,
                spec.head_dim,
                &ids,
                rows.get(&format!("{p}.weight")).unwrap().data,
                rows.get(&format!("{p}.scale")).unwrap().data,
            )
            .unwrap();
            (l.source_layer, Arc::new(table) as Arc<dyn EngramTable>)
        })
        .collect();
    let weights = DsWeights::load(
        store,
        config,
        &DsLoadOptions {
            skip_drafter: false,
            engram_tables,
        },
    )
    .unwrap();
    (
        Arc::new(DsModel::new(weights, Some(token_map)).unwrap()),
        tokens,
    )
}

/// The executor with every sampled token replaced by the known sequence's next token:
/// the engine then decodes that sequence one token a step, whatever the model predicts.
struct Forced {
    inner: DsCpuExecutor,
    truth: Vec<u32>,
}

impl Executor for Forced {
    fn spec(&self) -> &ModelSpec {
        self.inner.spec()
    }

    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
        let mut out = self.inner.execute(step)?;
        for (row, e) in step.seqs.iter().enumerate() {
            if e.sample {
                let p = (e.context_len + e.num_tokens - 1) as usize;
                out.tokens[row * out.stride as usize] = self.truth[p + 1];
                out.num_tokens[row] = 1;
            }
        }
        Ok(out)
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Decode rows on the released shape: a prompt of `n` tokens of the goldens' passage,
/// then that passage one token a step, each row's target logits against the model
/// crate's dense forward over the same tokens, bit for bit. One run decodes across
/// every parity, every block boundary (16), the window (128) and the pooled pairs.
#[test]
#[ignore = "needs a fetched truncation of the real checkpoint"]
fn released_decode_rows_equal_the_dense_forward() {
    let (model, tokens) = load();
    let c = model.config();
    let vocab = c.vocab_size;
    eprintln!(
        "{} tokens, {} layers, window {}, top-k {}",
        tokens.len(),
        c.layers.len(),
        c.attention.window,
        c.indexer.top_k
    );
    let dense = model
        .forward(
            &mut model.new_state(),
            &tokens,
            &DsForwardOptions::default(),
        )
        .unwrap()
        .logits;
    eprintln!("dense forward done");

    let mut failures = Vec::new();
    for (n, steps, width, chunk) in [
        (20usize, 236usize, 0u32, 1000u32),
        (33, 8, 0, 7),
        (130, 8, 0, 48),
    ] {
        let steps = steps.min(tokens.len() - n - 1);
        let cfg = DsCpuExecutorConfig {
            block_size: 16,
            num_blocks: 64,
            window_blocks: 64,
            host_blocks: 0,
            num_state_slots: 2,
            max_model_len: 1024,
            buckets: vec![Bucket {
                max_seqs: 2,
                max_tokens: 512,
            }],
            draft_width: width,
            sampleable_vocab_size: vocab as u32,
            record: true,
        };
        let exec = Forced {
            inner: DsCpuExecutor::new(model.clone(), cfg),
            truth: tokens.clone(),
        };
        let sched = SchedulerConfig {
            max_batched_tokens: 512,
            max_seqs: 2,
            max_prefill_chunk: chunk,
            eos_token_ids: Vec::new(),
            speculative: width > 0,
            cache: CachePolicy::default(),
            sweep_interval_ms: 1000,
        };
        let mut eng = Engine::new(exec, sched).unwrap();
        eng.submit(Request {
            id: 0,
            prompt: tokens[..n].to_vec(),
            sampling: SamplingParams::greedy(),
            max_tokens: steps as u32,
            stop_token_ids: Vec::new(),
            cache: CacheScope::Private,
            required_cached_tokens: 0,
        })
        .unwrap();
        let (mut now, mut checked, mut first) = (0, 0, None);
        while eng.unfinished() > 0 {
            eng.step(now).unwrap();
            now += 1;
            for r in eng.executor().inner.take_records() {
                for (pos, l) in &r.target_logits {
                    // Rows past the last host position are verified drafts, which are
                    // off the forced sequence.
                    if *pos >= r.context_len + r.num_tokens {
                        continue;
                    }
                    checked += 1;
                    let want = dense.row(*pos as usize);
                    if bits(l) != bits(want) && first.is_none() {
                        let differing = l.iter().zip(want).filter(|(a, b)| a != b).count();
                        let top = |x: &[f32]| {
                            (0..x.len()).max_by(|a, b| x[*a].total_cmp(&x[*b])).unwrap()
                        };
                        first = Some(format!(
                            "prompt {n} chunk {chunk} width {width}: logits at position {pos} \
                             (row of {} tokens from {}): {differing} of {vocab} differ; \
                             argmax {} against {}",
                            r.num_tokens,
                            r.context_len,
                            top(l),
                            top(want)
                        ));
                    }
                }
            }
        }
        eprintln!("prompt {n} chunk {chunk} width {width}: {checked} rows checked, {first:?}");
        failures.extend(first);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// An Engram table of pseudo-random rows, defined at every index: the same function for
/// the dense forward and the executor, so any token sequence can be compared.
struct AnyRow {
    rows: u64,
    dim: usize,
}

impl EngramTable for AnyRow {
    fn num_rows(&self) -> u64 {
        self.rows
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn read_row(&self, index: u64, out: &mut [f32]) -> eidola_engine_model::Result<()> {
        let mut z = index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5bf0_3635;
        for o in out.iter_mut() {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            *o = ((z >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.25;
            *o = f32::from_bits(o.to_bits() & 0xffff_0000);
        }
        Ok(())
    }
}

fn load_with_any_rows() -> Arc<DsModel> {
    let dir = PathBuf::from(std::env::var("EIDOLA_DEEPSEEK_TRUNCATED_DIR").unwrap());
    let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
    let pinned = open(&dir.join("golden/reference-pinned.safetensors"));
    let token_map = ints(&pinned, "engram.token_map")
        .iter()
        .map(|t| *t as u32)
        .collect();
    let config = config_of(&store)
        .unwrap()
        .truncated(&LAYERS)
        .unwrap()
        .with_routed_experts(BACKBONE_EXPERTS, DRAFTER_EXPERTS)
        .unwrap();
    let spec = config.engram.clone().unwrap();
    let engram_tables = spec
        .layers
        .iter()
        .filter(|l| LAYERS.contains(&l.source_layer))
        .map(|l| {
            let table = AnyRow {
                rows: l.num_rows,
                dim: spec.head_dim,
            };
            (l.source_layer, Arc::new(table) as Arc<dyn EngramTable>)
        })
        .collect();
    let weights = DsWeights::load(
        store,
        config,
        &DsLoadOptions {
            skip_drafter: false,
            engram_tables,
        },
    )
    .unwrap();
    Arc::new(DsModel::new(weights, Some(token_map)).unwrap())
}

const CHAT: [u32; 170] = [
    0, 128799, 77666, 288, 9504, 482, 28, 223, 2402, 343, 16801, 223, 19, 15, 1457, 14, 270, 4077,
    270, 1990, 14, 270, 850, 15146, 270, 22805, 868, 128803, 3085, 344, 270, 6102, 294, 9338, 33,
    9361, 295, 834, 10175, 16, 128804, 128821, 2581, 344, 13070, 16, 21017, 20407, 582, 3085, 4,
    344, 260, 34089, 34089, 16, 342, 1309, 1309, 582, 79666, 4, 344, 260, 34089, 34089, 16, 342,
    1309, 1309, 582, 79666, 4, 344, 260, 2940, 16, 342, 1309, 582, 51119, 4, 344, 260, 2940, 16,
    342, 2118, 582, 51119, 4, 344, 260, 2940, 16, 342, 2118, 582, 51119, 4, 344, 260, 2940, 16,
    342, 2118, 582, 79666, 4, 344, 260, 2940, 16, 342, 2118, 582, 51119, 4, 344, 260, 16, 342,
    2118, 582, 79666, 4, 344, 260, 16, 342, 2118, 582, 1805, 4, 344, 260, 16, 342, 2118, 582,
    60705, 4, 344, 260, 16, 342, 2118, 582, 79666, 4, 344, 260, 16, 342, 2118, 582, 60705, 4, 344,
    260, 16, 342, 2118, 582, 60705, 4, 344, 260, 16, 342,
];

/// A chat-format prompt (special tokens above every id the goldens' passage holds) and a
/// continuation, decoded one token a step and with five drafts a step.
#[test]
#[ignore = "needs a fetched truncation of the real checkpoint"]
fn released_chat_decode_rows_equal_the_dense_forward() {
    let model = load_with_any_rows();
    let tokens = CHAT.to_vec();
    let vocab = model.config().vocab_size;
    let dense = model
        .forward(
            &mut model.new_state(),
            &tokens,
            &DsForwardOptions::default(),
        )
        .unwrap()
        .logits;
    eprintln!("dense forward done");
    let mut failures = Vec::new();
    for (n, steps, width) in [(42usize, 100usize, 0u32), (42, 40, 5), (55, 20, 5)] {
        let cfg = DsCpuExecutorConfig {
            block_size: 16,
            num_blocks: 56,
            window_blocks: 56,
            host_blocks: 0,
            num_state_slots: 1,
            max_model_len: 192,
            buckets: vec![Bucket {
                max_seqs: 1,
                max_tokens: 64,
            }],
            draft_width: width,
            sampleable_vocab_size: 129_264,
            record: true,
        };
        let exec = Forced {
            inner: DsCpuExecutor::new(model.clone(), cfg),
            truth: tokens.clone(),
        };
        let sched = SchedulerConfig {
            max_batched_tokens: 64,
            max_seqs: 1,
            max_prefill_chunk: 64,
            eos_token_ids: Vec::new(),
            speculative: width > 0,
            cache: CachePolicy::default(),
            sweep_interval_ms: 1000,
        };
        let mut eng = Engine::new(exec, sched).unwrap();
        eng.submit(Request {
            id: 0,
            prompt: tokens[..n].to_vec(),
            sampling: SamplingParams::greedy(),
            max_tokens: steps as u32,
            stop_token_ids: Vec::new(),
            cache: CacheScope::Private,
            required_cached_tokens: 0,
        })
        .unwrap();
        let (mut now, mut checked, mut first) = (0, 0, None);
        while eng.unfinished() > 0 {
            eng.step(now).unwrap();
            now += 1;
            for r in eng.executor().inner.take_records() {
                for (pos, l) in &r.target_logits {
                    if *pos >= r.context_len + r.num_tokens {
                        continue;
                    }
                    checked += 1;
                    if bits(l) != bits(dense.row(*pos as usize)) && first.is_none() {
                        first = Some(format!("prompt {n} width {width}: logits at {pos} differ"));
                    }
                }
            }
        }
        eprintln!("prompt {n} width {width}: {checked} rows checked, {first:?} (vocab {vocab})");
        failures.extend(first);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn argmax(x: &[f32]) -> usize {
    (0..x.len()).max_by(|a, b| x[*a].total_cmp(&x[*b])).unwrap()
}

/// Generation from a chat prompt, three ways over the same tokens, printing what each
/// finds before asserting anything:
///
/// 1. the dense forward, teacher-forced along `tokens` (the prompt, then a recorded
///    continuation): where its greedy choice is the recorded next token;
/// 2. the executor through the engine, decoding `tokens` one token a step: every row's
///    logits against the dense forward's, and at the first row that differs the first
///    layer whose state differs;
/// 3. the dense forward generating greedily from the prompt by itself.
fn follow(model: Arc<DsModel>, tokens: &[u32], prompt: usize, steps: usize, sampleable: u32) {
    let c = model.config();
    let used = &tokens[..prompt + steps + 1];
    let opts = DsForwardOptions {
        taps: true,
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let dense = model.forward(&mut model.new_state(), used, &opts).unwrap();
    eprintln!(
        "1. dense forward over {} tokens ({} layers) in {:.0} s",
        used.len(),
        c.layers.len(),
        t.elapsed().as_secs_f64()
    );
    let choices: Vec<usize> = (prompt - 1..used.len() - 1)
        .map(|p| argmax(&dense.logits.row(p)[..sampleable as usize]))
        .collect();
    let same = choices
        .iter()
        .zip(&used[prompt..])
        .filter(|(a, b)| **a == **b as usize)
        .count();
    eprintln!(
        "   its greedy choice is the recorded next token at {same} of {} positions",
        choices.len()
    );
    eprintln!("   dense choices   {:?}", &choices[..choices.len().min(24)]);
    eprintln!(
        "   recorded tokens {:?}",
        &used[prompt..used.len().min(prompt + 24)]
    );

    let cfg = DsCpuExecutorConfig {
        block_size: 16,
        num_blocks: 56,
        window_blocks: 56,
        host_blocks: 0,
        num_state_slots: 1,
        max_model_len: 192,
        buckets: vec![Bucket {
            max_seqs: 1,
            max_tokens: 64,
        }],
        draft_width: 0,
        sampleable_vocab_size: sampleable,
        record: true,
    };
    let exec = Forced {
        inner: DsCpuExecutor::new(model.clone(), cfg),
        truth: used.to_vec(),
    };
    let sched = SchedulerConfig {
        max_batched_tokens: 64,
        max_seqs: 1,
        max_prefill_chunk: 64,
        eos_token_ids: Vec::new(),
        speculative: false,
        cache: CachePolicy::default(),
        sweep_interval_ms: 1000,
    };
    let mut eng = Engine::new(exec, sched).unwrap();
    eng.submit(Request {
        id: 0,
        prompt: used[..prompt].to_vec(),
        sampling: SamplingParams::greedy(),
        max_tokens: steps as u32,
        stop_token_ids: Vec::new(),
        cache: CacheScope::Private,
        required_cached_tokens: 0,
    })
    .unwrap();
    let t = std::time::Instant::now();
    let (mut now, mut rows, mut differing) = (0, 0, Vec::new());
    while eng.unfinished() > 0 {
        eng.step(now).unwrap();
        now += 1;
        for r in eng.executor().inner.take_records() {
            let p = (r.context_len + r.num_tokens - 1) as usize;
            let Some((_, l)) = r.target_logits.iter().find(|(pos, _)| *pos as usize == p) else {
                continue;
            };
            rows += 1;
            let want = dense.logits.row(p);
            let layers: Vec<usize> = (0..r.layer_streams.len())
                .filter(|layer| {
                    let got = r.layer_streams[*layer];
                    got != 0
                        && got
                            != eidola_engine_cpu::deepseek::stream_hash(dense.taps[*layer].row(p))
                })
                .collect();
            if bits(l) != bits(want) || !layers.is_empty() {
                let n = l.iter().zip(want).filter(|(a, b)| a != b).count();
                differing.push(format!(
                    "position {p} (a row of {} from {}): {n} logits differ, greedy choice {} \
                     against the dense forward's {}; layers whose state differs: first {:?}, {} in all",
                    r.num_tokens,
                    r.context_len,
                    argmax(&l[..sampleable as usize]),
                    argmax(&want[..sampleable as usize]),
                    layers.first(),
                    layers.len()
                ));
            }
        }
    }
    eprintln!(
        "2. the executor decoded {rows} rows in {:.0} s; {} differ from the dense forward",
        t.elapsed().as_secs_f64(),
        differing.len()
    );
    for d in differing.iter().take(12) {
        eprintln!("   {d}");
    }

    let t = std::time::Instant::now();
    let free = eidola_engine_cpu::deepseek::oracle::ds_dense_generate(
        &model,
        sampleable,
        &used[..prompt],
        &SamplingParams::greedy(),
        24.min(steps as u32),
        &[],
    )
    .unwrap();
    eprintln!(
        "3. the dense forward generating by itself ({:.0} s): {free:?}",
        t.elapsed().as_secs_f64()
    );
    assert!(differing.is_empty(), "the executor left the dense forward");
}

/// [`follow`] on the truncation, with a table defined at every row: the comparison's
/// own check, and the executor against the forward on chat tokens at released widths.
#[test]
#[ignore = "needs a fetched truncation of the real checkpoint"]
fn truncated_generation_follows_the_dense_forward() {
    follow(load_with_any_rows(), &CHAT, 42, 12, 129_264);
}

/// [`follow`] on the whole released checkpoint, with its own Engram tables: `CHAT` is a
/// prompt and what a run of the whole model generated after it.
///
///     EIDOLA_DEEPSEEK_DIR=<checkpoint dir> \
///     EIDOLA_DEEPSEEK_TOKEN_MAP=<a safetensors file holding engram.token_map> \
///         cargo test --release -p eidola-engine-cpu --test deepseek_real -- \
///         --ignored --nocapture released_generation
#[test]
#[ignore = "needs the whole released checkpoint"]
fn released_generation_follows_the_dense_forward() {
    let dir = PathBuf::from(
        std::env::var("EIDOLA_DEEPSEEK_DIR").expect("EIDOLA_DEEPSEEK_DIR names the checkpoint"),
    );
    let map = PathBuf::from(
        std::env::var("EIDOLA_DEEPSEEK_TOKEN_MAP")
            .expect("EIDOLA_DEEPSEEK_TOKEN_MAP names a file holding engram.token_map"),
    );
    let token_map = ints(&open(&map), "engram.token_map")
        .iter()
        .map(|t| *t as u32)
        .collect();
    let t = std::time::Instant::now();
    let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
    let config = config_of(&store).unwrap();
    let weights = DsWeights::load(store, config, &DsLoadOptions::default()).unwrap();
    let model = Arc::new(DsModel::new(weights, Some(token_map)).unwrap());
    eprintln!(
        "loaded the whole model in {:.0} s",
        t.elapsed().as_secs_f64()
    );
    let steps = std::env::var("EIDOLA_DEEPSEEK_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    follow(model, &CHAT, 42, steps, 129_264);
}
