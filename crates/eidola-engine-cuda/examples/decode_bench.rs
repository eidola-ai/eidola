//! Decode step latency per rung of the decode-graph ladder, on a MiMo-V2.6
//! checkpoint: the eager path, the padded launches run directly, and the
//! captured graph replayed, on one executor.
//!
//! ```text
//! decode_bench <kernels_dir> <model_dir> [--layers all|<i,j,...>] [--context N]
//!              [--iters N] [--max-rows N]
//! ```
//!
//! Each rung runs a full batch (`rows` = the rung) of decode rows, every row at
//! `--context` positions (default 1024) of zero KV mapped through its own
//! blocks (a sliding group only the blocks its window reaches), so attention
//! reads realistic page lists; tokens are fixed ids, and
//! sampling is greedy. Per rung and path: `--iters` timed steps (default 50)
//! after 5 warm-up steps, each step's wall time measured around
//! `Executor::execute` (the host work, maintenance, upload, launch or replay,
//! and the token read-back). Prints a table and one JSON line per rung and
//! path (`{"rung", "path", "median_ms", "p90_ms", "mean_ms"}`), plus each
//! rung's capture memory. `--max-rows` (default 64) is the largest rung.
//! Every path rewrites the same positions, so each starts from the same KV.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::executor::{Executor, SeqEntry, StepInput, TableUpdate};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::{AttentionKind, Bucket};
use eidola_engine_cuda::{
    CudaExecutor, CudaExecutorConfig, CudaGraphs, DecodePath, Gpu, KernelDir, KvBlocks,
};
use eidola_engine_model::safetensors::WeightSet;

const BS: u32 = 16;
const SAMPLEABLE: u32 = 151_675;
const WARMUP: usize = 5;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|x| x == name)?;
    Some(
        args.get(i + 1)
            .unwrap_or_else(|| panic!("{name} takes a value")),
    )
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
        Some(list) => Some(list.split(',').map(|x| x.parse().unwrap()).collect()),
    };
    let context: u32 = flag(rest, "--context").map_or(1024, |x| x.parse().unwrap());
    let iters: usize = flag(rest, "--iters").map_or(50, |x| x.parse().unwrap());
    let max_rows: u32 = flag(rest, "--max-rows").map_or(64, |x| x.parse().unwrap());
    assert!(context >= 1 && iters >= 1 && max_rows >= 1);

    // Positions: the context, then one per warm-up and timed step.
    let steps = u32::try_from(WARMUP + iters).unwrap();
    let last = context + steps;
    let max_len = (last + 1).next_multiple_of(BS);
    let store = Arc::new(WeightSet::open_dir(Path::new(model)).unwrap());
    let mut cfg = CudaExecutorConfig {
        block_size: BS,
        num_blocks: KvBlocks {
            global: 2,
            sliding: 2,
        },
        num_state_slots: max_rows,
        max_model_len: max_len,
        buckets: vec![Bucket {
            max_seqs: max_rows,
            max_tokens: max_rows.max(512),
        }],
        sampleable_vocab_size: SAMPLEABLE,
        image: None,
        graphs: CudaGraphs::On,
    };
    // Per group, the logical blocks a row reads or writes over the run.
    let spec = CudaExecutor::preflight(&store, layers.as_deref(), &cfg).unwrap();
    let logical: Vec<(u32, u32)> = spec
        .kv_groups
        .iter()
        .map(|g| (g.attention.first_visible(context) / BS, last / BS))
        .collect();
    let per_slot: Vec<u32> = logical.iter().map(|(a, b)| b - a + 1).collect();
    let pool = |kind: AttentionKind| -> u32 {
        spec.kv_groups
            .iter()
            .zip(&per_slot)
            .find(|(g, _)| std::mem::discriminant(&g.attention) == std::mem::discriminant(&kind))
            .map_or(0, |(_, n)| 1 + max_rows * n)
    };
    cfg.num_blocks = KvBlocks {
        global: pool(AttentionKind::Full),
        sliding: pool(AttentionKind::Sliding { window: 1 }),
    };
    let gpu = Gpu::open(0).unwrap();
    let t0 = Instant::now();
    let mut ex = CudaExecutor::new(gpu, &KernelDir::new(kernels), store, layers.as_deref(), cfg)
        .expect("executor with every rung captured");
    let graphs = ex.decode_graphs().unwrap();
    let ladder = graphs.ladder().to_vec();
    println!("loaded and captured in {:.1?}", t0.elapsed());
    for (r, bytes) in ladder.iter().zip(graphs.capture_bytes()) {
        println!("rung {r:>4}: capture took {bytes} bytes of device memory");
    }

    // Every slot's blocks, in both groups, mapped once.
    let mut updates = Vec::new();
    for (group, (&(first, end), &n)) in (0u32..).zip(logical.iter().zip(&per_slot)) {
        for slot in 0..max_rows {
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
    let decode = |rows: u32, pos: u32, updates: Vec<TableUpdate>| StepInput {
        bucket: Bucket {
            max_seqs: max_rows,
            max_tokens: max_rows.max(512),
        },
        maintenance: vec![],
        table_updates: updates,
        seqs: (0..rows)
            .map(|slot| SeqEntry {
                slot,
                token_start: slot,
                num_tokens: 1,
                context_len: pos,
                num_drafts: 0,
                sample: true,
                sampling: SamplingParams::greedy(),
            })
            .collect(),
        token_ids: (0..rows).map(|i| 1000 + i).collect(),
        positions: vec![pos; rows as usize],
        return_logits: false,
    };
    ex.set_decode_path(DecodePath::Eager);
    ex.execute(&decode(1, context, updates)).unwrap();

    println!(
        "\n{:>5} {:>8} {:>11} {:>11} {:>11}",
        "rung", "path", "median ms", "p90 ms", "mean ms"
    );
    let mut json = Vec::new();
    for &rung in &ladder {
        let mut medians = Vec::new();
        for path in [DecodePath::Eager, DecodePath::Direct, DecodePath::Replay] {
            ex.set_decode_path(path);
            let mut times = Vec::with_capacity(iters);
            for i in 0..WARMUP + iters {
                let step = decode(rung, context + u32::try_from(i).unwrap(), vec![]);
                let t = Instant::now();
                ex.execute(&step).unwrap();
                if i >= WARMUP {
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            times.sort_by(f64::total_cmp);
            let median = times[times.len() / 2];
            let p90 = times[(times.len() * 9 / 10).min(times.len() - 1)];
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            println!("{rung:>5} {path:>8?} {median:>11.3} {p90:>11.3} {mean:>11.3}");
            json.push(format!(
                r#"{{"rung":{rung},"path":"{path:?}","median_ms":{median:.4},"p90_ms":{p90:.4},"mean_ms":{mean:.4}}}"#
            ));
            medians.push(median);
        }
        println!(
            "      replay is {:.2}x eager, {:.2}x direct",
            medians[0] / medians[2],
            medians[1] / medians[2]
        );
    }
    println!();
    for line in json {
        println!("{line}");
    }
    println!("{:?}", ex.decode_stats());
}
