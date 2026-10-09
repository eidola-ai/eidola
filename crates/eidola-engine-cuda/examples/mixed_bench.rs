//! Mixed step latency on the mixed-step graph ladder, on a MiMo-V2.6
//! checkpoint: the eager path, the padded launches run directly, and the
//! captured graph replayed, on one executor.
//!
//! ```text
//! mixed_bench <kernels_dir> <model_dir> [--layers all|<i,j,...>] [--context N]
//!             [--iters N] [--rows <r,...>] [--extend <e,...>]
//! ```
//!
//! Each shape is `rows` decode rows (one token each) beside one extend row of
//! `extend` tokens (a continuation, as a tool loop's next turn arrives), every
//! row at `--context` positions (default 1024) of zero KV mapped through its
//! own blocks, so attention reads realistic page lists; tokens are fixed ids,
//! every row samples greedily. Shapes are every pair of `--rows` (default
//! `0,1,8,32,63`) and `--extend` (default `16,64,128,256,364,448`) whose
//! tokens fit the 512-token bucket. Per shape and path: `--iters` timed steps
//! (default 30) after 3 warm-up steps, each step's wall time measured around
//! `Executor::execute` (the host work, maintenance, upload, launch or replay,
//! and the read-back); every step rewrites the same positions, so each path
//! starts from the same KV. Prints a table and one JSON line per shape and
//! path (`{"rows", "extend", "tokens", "rung", "path", "median_ms",
//! "p90_ms", "mean_ms"}`; `rung` is 0 when no rung holds the step, which
//! then runs eagerly on every path), plus each mixed rung's capture memory.
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::executor::{Executor, SeqEntry, StepInput, TableUpdate};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::{AttentionKind, Bucket, KvRole};
use eidola_engine_cuda::mixed::mixed_rung;
use eidola_engine_cuda::{
    CudaExecutor, CudaExecutorConfig, CudaGraphs, DecodePath, Gpu, KernelDir, KvBlocks, MtpHidden,
};
use eidola_engine_model::safetensors::WeightSet;

const BS: u32 = 16;
const SAMPLEABLE: u32 = 151_675;
const WARMUP: usize = 3;
const MAX_TOKENS: u32 = 512;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|x| x == name)?;
    Some(
        args.get(i + 1)
            .unwrap_or_else(|| panic!("{name} takes a value")),
    )
}

fn list(s: &str) -> Vec<u32> {
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (kernels, model) = match (a.get(1), a.get(2)) {
        (Some(k), Some(m)) => (k.as_str(), m.as_str()),
        _ => {
            eprintln!("usage: see the module docs");
            std::process::exit(2);
        }
    };
    let rest = &a[3..];
    let layers: Option<Vec<usize>> = match flag(rest, "--layers") {
        None | Some("all") => None,
        Some(l) => Some(l.split(',').map(|x| x.parse().unwrap()).collect()),
    };
    let context: u32 = flag(rest, "--context").map_or(1024, |x| x.parse().unwrap());
    let iters: usize = flag(rest, "--iters").map_or(30, |x| x.parse().unwrap());
    let rows_list = flag(rest, "--rows").map_or_else(|| vec![0, 1, 8, 32, 63], list);
    let extend_list = flag(rest, "--extend").map_or_else(|| vec![16, 64, 128, 256, 364, 448], list);
    let shapes: Vec<(u32, u32)> = rows_list
        .iter()
        .flat_map(|&r| extend_list.iter().map(move |&e| (r, e)))
        .filter(|&(r, e)| e >= 1 && r + e <= MAX_TOKENS)
        .collect();
    assert!(context >= 1 && iters >= 1 && !shapes.is_empty());
    let max_rows = rows_list.iter().copied().max().unwrap();
    let max_extend = extend_list.iter().copied().max().unwrap();
    // Decode rows in slots `0..rows`, the extend row in slot `max_rows`.
    let slots = max_rows + 1;
    let last = context + max_extend;
    let max_len = (last + 1).next_multiple_of(BS);
    let store = Arc::new(WeightSet::open_dir(Path::new(model)).unwrap());
    let bucket = Bucket {
        max_seqs: slots,
        max_tokens: MAX_TOKENS,
    };
    let mut cfg = CudaExecutorConfig {
        block_size: BS,
        num_blocks: KvBlocks {
            global: 2,
            sliding: 2,
            drafter: 0,
        },
        num_state_slots: slots,
        max_model_len: max_len,
        buckets: vec![bucket],
        sampleable_vocab_size: SAMPLEABLE,
        image: None,
        graphs: CudaGraphs::On,
        draft_tokens: 0,
        mtp_hidden: MtpHidden::Normed,
    };
    // Per group, the logical blocks a row reads or writes.
    let spec = CudaExecutor::preflight(&store, layers.as_deref(), &cfg).unwrap();
    let logical: Vec<(u32, u32)> = spec
        .kv_groups
        .iter()
        .map(|g| (g.attention.first_visible(context) / BS, last / BS))
        .collect();
    let per_slot: Vec<u32> = logical.iter().map(|(a, b)| b - a + 1).collect();
    let pool = |pick: &dyn Fn(&eidola_engine::spec::KvGroupSpec) -> bool| -> u32 {
        spec.kv_groups
            .iter()
            .zip(&per_slot)
            .find(|(g, _)| pick(g))
            .map_or(0, |(_, n)| 1 + slots * n)
    };
    cfg.num_blocks = KvBlocks {
        global: pool(&|g| g.attention == AttentionKind::Full),
        sliding: pool(&|g| g.role == KvRole::Target && g.attention != AttentionKind::Full),
        drafter: 0,
    };
    let gpu = Gpu::open(0).unwrap();
    let t0 = Instant::now();
    let mut ex = CudaExecutor::new(gpu, &KernelDir::new(kernels), store, layers.as_deref(), cfg)
        .expect("executor with every rung captured");
    println!("loaded and captured in {:.1?}", t0.elapsed());
    let mixed = ex.mixed_graphs().expect("graphs on");
    let ladder = mixed.ladder().to_vec();
    for (r, bytes) in ladder.iter().zip(mixed.capture_bytes()) {
        println!("mixed rung {r:>4}: capture took {bytes} bytes of device memory");
    }
    println!(
        "mixed captures {} bytes in all; step table and read-back {} bytes",
        mixed.capture_bytes().iter().sum::<i64>(),
        mixed.table_bytes()
    );

    // Every slot's blocks, in every group, mapped once.
    let mut updates = Vec::new();
    for (group, (&(first, end), &n)) in (0u32..).zip(logical.iter().zip(&per_slot)) {
        for slot in 0..slots {
            for (k, index) in (0u32..).zip(first..=end) {
                updates.push(TableUpdate {
                    slot,
                    group,
                    index,
                    block: 1 + slot * n + k,
                });
            }
        }
    }
    let step = |rows: u32, extend: u32, updates: Vec<TableUpdate>| {
        let mut seqs: Vec<SeqEntry> = (0..rows)
            .map(|slot| SeqEntry {
                slot,
                token_start: slot,
                num_tokens: 1,
                context_len: context,
                num_drafts: 0,
                sample: true,
                sampling: SamplingParams::greedy(),
            })
            .collect();
        seqs.push(SeqEntry {
            slot: max_rows,
            token_start: rows,
            num_tokens: extend,
            context_len: context,
            num_drafts: 0,
            sample: true,
            sampling: SamplingParams::greedy(),
        });
        let mut positions = vec![context; rows as usize];
        positions.extend(context..context + extend);
        StepInput {
            bucket,
            maintenance: vec![],
            table_updates: updates,
            seqs,
            token_ids: (0..rows + extend).map(|i| 1000 + i).collect(),
            positions,
            return_logits: false,
        }
    };
    ex.set_decode_path(DecodePath::Eager);
    ex.execute(&step(0, 1, updates)).unwrap();

    println!(
        "\n{:>5} {:>6} {:>6} {:>5} {:>8} {:>11} {:>11} {:>11}",
        "rows", "extend", "tokens", "rung", "path", "median ms", "p90 ms", "mean ms"
    );
    let mut json = Vec::new();
    for &(rows, extend) in &shapes {
        let tokens = rows + extend;
        // Flash's groups tile a row of nine or more tokens at 128.
        let tile = if extend >= 9 { 128 } else { 0 };
        let rung = mixed_rung(&ladder, tokens as usize, &[tile, tile]).map_or(0, |r| ladder[r]);
        let input = step(rows, extend, vec![]);
        let mut medians = Vec::new();
        for path in [DecodePath::Eager, DecodePath::Direct, DecodePath::Replay] {
            ex.set_decode_path(path);
            let mut times = Vec::with_capacity(iters);
            for i in 0..WARMUP + iters {
                let t = Instant::now();
                ex.execute(&input).unwrap();
                if i >= WARMUP {
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            times.sort_by(f64::total_cmp);
            let median = times[times.len() / 2];
            let p90 = times[(times.len() * 9 / 10).min(times.len() - 1)];
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            println!(
                "{rows:>5} {extend:>6} {tokens:>6} {rung:>5} {path:>8?} {median:>11.3} {p90:>11.3} \
                 {mean:>11.3}"
            );
            json.push(format!(
                r#"{{"rows":{rows},"extend":{extend},"tokens":{tokens},"rung":{rung},"path":"{path:?}","median_ms":{median:.4},"p90_ms":{p90:.4},"mean_ms":{mean:.4}}}"#
            ));
            medians.push(median);
        }
        println!(
            "      replay is {:.2}x eager ({:+.3} ms), {:.2}x direct",
            medians[0] / medians[2],
            medians[2] - medians[0],
            medians[1] / medians[2]
        );
    }
    println!();
    for line in json {
        println!("{line}");
    }
    println!("{:?}", ex.decode_stats());
}
