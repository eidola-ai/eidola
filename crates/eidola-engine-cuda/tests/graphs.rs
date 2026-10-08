//! Decode-graph replay on a real MiMo-V2.6-Flash checkpoint (the truncation,
//! checkpoint layers 0, 1, 2 and 5, by default; every layer with
//! `EIDOLA_MIMO_LAYERS=all`):
//!
//! - every rung of the ladder captures at construction;
//! - on a mixed set of decode rows (contexts from 1 to past the sliding
//!   window, greedy and seeded rows, every rung), each decode step's tokens
//!   and logits are bit-identical whether the captured graph replays, its
//!   launches run directly, or the eager path runs; and padding writes no
//!   block outside the pad blocks;
//! - through the engine, a workload mixing prefill chunks and decode steps
//!   produces the same tokens with graphs replaying as eagerly, and every
//!   free block stays zero or queued for zeroing after every step.
//!
//! Needs a GPU, `EIDOLA_ENGINE_KERNELS_DIR`, `EIDOLA_MIMO_DIR` and
//! `EIDOLA_MIMO_GOLDEN` (as `real_flash.rs`); prints what it measured, so run
//! with `--nocapture`.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use common::setup;
use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, SeqEntry, StepInput, TableUpdate};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_cuda::{CudaExecutor, CudaExecutorConfig, CudaGraphs, DecodePath, KvBlocks};
use eidola_engine_model::safetensors::WeightSet;

const KEEP: [usize; 4] = [0, 1, 2, 5];
const BS: u32 = 16;
const SAMPLEABLE: u32 = 151_675;
/// Blocks per slot in the seam-level test (512 positions).
const PER_SLOT: u32 = 32;
const SLOTS: u32 = 8;

fn layers() -> Option<Vec<usize>> {
    match std::env::var("EIDOLA_MIMO_LAYERS").as_deref() {
        Ok("all") => None,
        _ => Some(KEEP.to_vec()),
    }
}

struct Fixture {
    store: Arc<WeightSet>,
    tokens: Vec<u32>,
}

impl Fixture {
    /// `len` of the golden's real token ids from `start`, wrapping around.
    fn text(&self, start: usize, len: usize) -> Vec<u32> {
        (start..start + len)
            .map(|i| self.tokens[i % self.tokens.len()])
            .collect()
    }
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
        .map(|x| u32::try_from(x).unwrap())
        .collect();
    Some(Fixture {
        store: Arc::new(WeightSet::open_dir(&PathBuf::from(dir)).unwrap()),
        tokens,
    })
}

fn executor(fx: &Fixture, blocks: u32, buckets: Vec<Bucket>) -> Option<CudaExecutor> {
    let su = setup()?;
    let cfg = CudaExecutorConfig {
        block_size: BS,
        num_blocks: KvBlocks {
            global: blocks,
            sliding: blocks,
        },
        num_state_slots: SLOTS,
        max_model_len: 1024,
        buckets,
        sampleable_vocab_size: SAMPLEABLE,
        image: None,
        graphs: CudaGraphs::On,
    };
    let t0 = std::time::Instant::now();
    let ex = CudaExecutor::new(su.gpu, &su.dir, fx.store.clone(), layers().as_deref(), cfg)
        .expect("every rung captures");
    let graphs = ex.decode_graphs().expect("graphs on");
    println!(
        "loaded and captured in {:.1?}: rungs {:?}, capture bytes {:?}",
        t0.elapsed(),
        graphs.ladder(),
        graphs.capture_bytes()
    );
    Some(ex)
}

/// Seeded sampling for odd slots, greedy for even ones.
fn sampling(slot: u32) -> SamplingParams {
    if slot.is_multiple_of(2) {
        SamplingParams::greedy()
    } else {
        SamplingParams::new(0.8, 40, 0.95, 0.0, 1000 + u64::from(slot)).unwrap()
    }
}

fn entry(slot: u32, start: u32, context: u32, n: u32) -> SeqEntry {
    SeqEntry {
        slot,
        token_start: start,
        num_tokens: n,
        context_len: context,
        num_drafts: 0,
        sample: true,
        sampling: sampling(slot),
    }
}

fn step_of(seqs: Vec<SeqEntry>, tokens: Vec<u32>, updates: Vec<TableUpdate>) -> StepInput {
    let positions = seqs
        .iter()
        .flat_map(|s| s.context_len..s.context_len + s.num_tokens)
        .collect();
    StepInput {
        bucket: Bucket {
            max_seqs: SLOTS,
            max_tokens: 512,
        },
        maintenance: vec![],
        table_updates: updates,
        seqs,
        token_ids: tokens,
        positions,
        return_logits: true,
    }
}

/// `slot`'s logical blocks `0..PER_SLOT` mapped to its own range in both
/// groups.
fn map(slot: u32) -> Vec<TableUpdate> {
    (0..2)
        .flat_map(|group| {
            (0..PER_SLOT).map(move |i| TableUpdate {
                slot,
                group,
                index: i,
                block: 1 + slot * PER_SLOT + i,
            })
        })
        .collect()
}

fn logit_bits(out: &eidola_engine::executor::StepOutput) -> Vec<Vec<u32>> {
    out.logits
        .as_ref()
        .unwrap()
        .iter()
        .map(|r| r[0].iter().map(|x| x.to_bits()).collect())
        .collect()
}

fn max_diff(a: &[u32], b: &[u32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f32::from_bits(x) - f32::from_bits(y)).abs())
        .fold(0.0, f32::max)
}

#[test]
fn decode_steps_replay_bit_for_bit() {
    let Some(fx) = fixture() else { return };
    let blocks = 1 + SLOTS * PER_SLOT + 16;
    let Some(mut ex) = executor(
        &fx,
        blocks,
        vec![Bucket {
            max_seqs: SLOTS,
            max_tokens: 512,
        }],
    ) else {
        return;
    };
    assert_eq!(ex.decode_graphs().unwrap().ladder(), [1, 2, 4, 8]);

    // Prefill eight slots eagerly: contexts from one token to past the
    // sliding window (128), from the golden's real token ids.
    let lens = [40u32, 17, 1, 130, 64, 5, 150, 2];
    assert!(lens.iter().sum::<u32>() <= 512, "one prefill step");
    let mut ctx = [0u32; SLOTS as usize];
    let mut next = [0u32; SLOTS as usize];
    let (mut seqs, mut tokens, mut updates) = (Vec::new(), Vec::new(), Vec::new());
    let mut at = 0usize;
    for (slot, &len) in (0u32..).zip(&lens) {
        seqs.push(entry(slot, u32::try_from(tokens.len()).unwrap(), 0, len));
        tokens.extend(fx.text(at, len as usize));
        at += len as usize;
        updates.extend(map(slot));
    }
    let out = ex.execute(&step_of(seqs, tokens, updates)).unwrap();
    for slot in 0..SLOTS as usize {
        ctx[slot] = lens[slot];
        next[slot] = out.row(slot)[0];
    }
    let eager_before = ex.decode_stats().eager;

    // Decode steps over every rung (8, 4, 2 and 1 rows, and partial rungs):
    // each step runs as replay, then directly, then eagerly, the same input
    // each time (KV for the step's position is rewritten with what the next
    // path computes, so each comparison starts from identical KV).
    let patterns: [&[u32]; 8] = [
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[0, 1, 2, 3, 4],
        &[3, 6, 7],
        &[1, 2],
        &[6],
        &[0, 2, 4, 6, 7],
        &[5, 7],
        &[0, 1, 2, 3, 4, 5, 6],
    ];
    let mut steps = 0u64;
    for round in 0..3 {
        for rows in patterns {
            let seqs: Vec<SeqEntry> = (0u32..)
                .zip(rows)
                .map(|(i, &s)| entry(s, i, ctx[s as usize], 1))
                .collect();
            let tokens: Vec<u32> = rows.iter().map(|&s| next[s as usize]).collect();
            let input = step_of(seqs, tokens, vec![]);
            let mut outs = Vec::new();
            for path in [DecodePath::Replay, DecodePath::Direct, DecodePath::Eager] {
                ex.set_decode_path(path);
                outs.push((path, ex.execute(&input).unwrap()));
            }
            let (_, replay) = &outs[0];
            let want = logit_bits(replay);
            for (path, out) in &outs[1..] {
                assert_eq!(
                    out.tokens, replay.tokens,
                    "round {round} rows {rows:?}: {path:?} tokens"
                );
                for (i, (got, want)) in logit_bits(out).iter().zip(&want).enumerate() {
                    assert!(
                        got == want,
                        "round {round} rows {rows:?}: row {i} logits differ on {path:?}, max |Δ| {}",
                        max_diff(got, want)
                    );
                }
            }
            for (i, &s) in rows.iter().enumerate() {
                ctx[s as usize] += 1;
                next[s as usize] = replay.row(i)[0];
                assert!(next[s as usize] < SAMPLEABLE);
            }
            steps += 1;
        }
    }
    let stats = ex.decode_stats();
    assert_eq!((stats.replayed, stats.direct), (steps, steps));
    assert_eq!(stats.eager, eager_before + steps);
    println!("{steps} decode steps, bit-identical on every path: {stats:?}");

    // Padding wrote no block the seam can name: every block no slot maps is
    // still zero.
    for g in 0..2 {
        for b in 1 + SLOTS * PER_SLOT..blocks {
            assert!(
                ex.block_is_zero(g, b).unwrap(),
                "group {g} block {b} written"
            );
        }
    }
}

fn request(id: u64, prompt: Vec<u32>, sampling: SamplingParams, max_tokens: u32) -> Request {
    Request {
        id,
        prompt,
        sampling,
        max_tokens,
        stop_token_ids: Vec::new(),
        cache: CacheScope::Private,
    }
}

/// Run `requests` (each submitted at its step) to completion with decode
/// steps on `path`; their outputs by index.
fn run_workload(
    eng: &mut Engine<CudaExecutor>,
    path: DecodePath,
    requests: &[(u64, Vec<u32>, SamplingParams, u32)],
    id_base: u64,
) -> Vec<Vec<u32>> {
    eng.executor().set_decode_path(path);
    let mut outputs: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut now = 0u64;
    let mut submitted = 0;
    let mut guard = 0;
    while submitted < requests.len() || eng.unfinished() > 0 {
        while submitted < requests.len() && requests[submitted].0 <= now {
            let (_, prompt, sampling, max) = &requests[submitted];
            let id = id_base + submitted as u64;
            outputs.insert(id, Vec::new());
            eng.submit(request(id, prompt.clone(), *sampling, *max))
                .unwrap();
            submitted += 1;
        }
        for e in eng.step(now).unwrap() {
            outputs.entry(e.id).or_default().extend(&e.tokens);
        }
        // Every free block is zero on the device or has a zero queued.
        let kv = eng.kv();
        kv.check_invariants();
        let pending = kv.pending_zeros();
        for (g, b) in kv.free_block_set() {
            if !pending.contains(&(g, b)) {
                assert!(
                    eng.executor().block_is_zero(g, b).unwrap(),
                    "free block {g}/{b} holds data with no zero pending"
                );
            }
        }
        now += 1;
        guard += 1;
        assert!(guard < 10_000, "no progress");
    }
    (0..requests.len())
        .map(|i| outputs.remove(&(id_base + i as u64)).unwrap())
        .collect()
}

#[test]
fn the_engine_serves_the_same_tokens_with_graphs() {
    let Some(fx) = fixture() else { return };
    let Some(ex) = executor(
        &fx,
        160,
        vec![
            Bucket {
                max_seqs: 1,
                max_tokens: 16,
            },
            Bucket {
                max_seqs: 6,
                max_tokens: 128,
            },
        ],
    ) else {
        return;
    };
    let mut eng = Engine::new(
        ex,
        SchedulerConfig {
            max_batched_tokens: 128,
            max_seqs: 6,
            max_prefill_chunk: 48,
            eos_token_ids: Vec::new(),
            speculative: false,
            cache: CachePolicy::default(),
            sweep_interval_ms: 1000,
        },
    )
    .unwrap();
    // Arrivals spread over the run, so steps mix chunked prefill with decode
    // and decode steps take every rung; greedy and seeded rows.
    let requests: Vec<(u64, Vec<u32>, SamplingParams, u32)> = (0..9u32)
        .map(|i| {
            let start = (i as usize * 23) % 120;
            let len = [3usize, 70, 12, 140, 1, 33, 90, 7, 50][i as usize];
            (
                u64::from(i) * 4,
                fx.text(start, len),
                sampling(i),
                [24u32, 16, 40, 8, 30, 20, 12, 36, 18][i as usize],
            )
        })
        .collect();
    let eager = run_workload(&mut eng, DecodePath::Eager, &requests, 0);
    let before = eng.executor().decode_stats();
    let replayed = run_workload(&mut eng, DecodePath::Replay, &requests, 1000);
    let after = eng.executor().decode_stats();
    for (i, (a, b)) in eager.iter().zip(&replayed).enumerate() {
        assert_eq!(a, b, "request {i}: eager and replayed outputs differ");
        assert_eq!(
            a.len(),
            requests[i].3 as usize,
            "request {i} ran to its length"
        );
    }
    assert!(after.replayed > before.replayed, "no decode step replayed");
    println!(
        "engine workload: eager run {before:?}, replayed run {} replays, {} eager steps",
        after.replayed - before.replayed,
        after.eager - before.eager
    );
}
