//! Decode step latency per rung of the decode-graph ladder, on a MiMo-V2.6
//! checkpoint: the eager path, the padded launches run directly, and the
//! captured graph replayed, on one executor.
//!
//! ```text
//! decode_bench <kernels_dir> <model_dir> [--layers all|<i,j,...>] [--context N]
//!              [--iters N] [--max-rows N] [--draft-tokens K] [--prompt <prompts.jsonl>]
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
//!
//! With `--draft-tokens K` (1 to 3; 0, the default, is the run above) the
//! executor drafts `K` tokens a step with the checkpoint's MTP layers, every
//! row of every step a uniform drafted decode row (the drafted ladder's
//! rungs), and each row continues from the tokens it produced; the table and
//! the JSON lines (`"draft_tokens"`, `"tokens_per_step"`: tokens a row
//! produced per step, `1 +` its accepted drafts) say how many tokens a step
//! yields. Over zero KV the drafts mean nothing; `--prompt` (the first row of
//! an `eval` prompts file, cut to whole blocks) prefills every row with real
//! text first, so acceptance is the model's. Each path starts every row at
//! the same position, from the same KV and taps.
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::executor::{Executor, SeqEntry, StepInput, TableUpdate};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::{AttentionKind, Bucket, KvRole};
use eidola_engine_cuda::{
    CudaExecutor, CudaExecutorConfig, CudaGraphs, DecodePath, Gpu, KernelDir, KvBlocks, MtpHidden,
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
    let depths: u32 = flag(rest, "--draft-tokens").map_or(0, |x| x.parse().unwrap());
    let prompt: Option<Vec<u32>> = flag(rest, "--prompt").map(|path| {
        let line = std::fs::read_to_string(path).unwrap();
        let first: serde_json::Value =
            serde_json::from_str(line.lines().next().expect("a prompt")).unwrap();
        let ids: Vec<u32> = first["prompt_ids"]
            .as_array()
            .expect("prompt_ids")
            .iter()
            .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
            .collect();
        // Whole blocks, and the token after them to decode from.
        let whole = (ids.len() - 1) / BS as usize * BS as usize;
        assert!(whole >= BS as usize, "a prompt of at least one block");
        ids[..=whole].to_vec()
    });
    let context = prompt
        .as_ref()
        .map_or(context, |p| u32::try_from(p.len() - 1).unwrap());
    assert!(context >= 1 && iters >= 1 && max_rows >= 1);
    assert!(depths <= 3, "--draft-tokens is at most 3");
    assert!(
        depths == 0 || context.is_multiple_of(BS),
        "a drafted run starts every path on a block boundary (its drafter loads the tap)"
    );

    // Positions: the context, then up to `1 + K` per warm-up and timed step.
    let steps = u32::try_from(WARMUP + iters).unwrap();
    let last = context + steps * (depths + 1);
    let max_len = (last + 1).next_multiple_of(BS);
    let max_tokens = (max_rows * (depths + 1)).max(512);
    let store = Arc::new(WeightSet::open_dir(Path::new(model)).unwrap());
    let mut cfg = CudaExecutorConfig {
        block_size: BS,
        num_blocks: KvBlocks {
            global: 2,
            sliding: 2,
            drafter: if depths > 0 { 2 } else { 0 },
        },
        num_state_slots: max_rows,
        max_model_len: max_len,
        buckets: vec![Bucket {
            max_seqs: max_rows,
            max_tokens,
        }],
        sampleable_vocab_size: SAMPLEABLE,
        image: None,
        graphs: CudaGraphs::On,
        draft_tokens: depths,
        mtp_hidden: MtpHidden::Normed,
    };
    // Per group, the logical blocks a row reads or writes over the run (from
    // position 0 when a prompt is prefilled).
    let spec = CudaExecutor::preflight(&store, layers.as_deref(), &cfg).unwrap();
    let logical: Vec<(u32, u32)> = spec
        .kv_groups
        .iter()
        .map(|g| {
            let first = if prompt.is_some() {
                0
            } else {
                g.attention.first_visible(context) / BS
            };
            (first, last / BS)
        })
        .collect();
    let per_slot: Vec<u32> = logical.iter().map(|(a, b)| b - a + 1).collect();
    let pool = |pick: &dyn Fn(&eidola_engine::spec::KvGroupSpec) -> bool| -> u32 {
        spec.kv_groups
            .iter()
            .zip(&per_slot)
            .find(|(g, _)| pick(g))
            .map_or(0, |(_, n)| 1 + max_rows * n)
    };
    cfg.num_blocks = KvBlocks {
        global: pool(&|g| g.attention == AttentionKind::Full),
        sliding: pool(&|g| g.role == KvRole::Target && g.attention != AttentionKind::Full),
        drafter: pool(&|g| g.role == KvRole::Drafter),
    };
    let gpu = Gpu::open(0).unwrap();
    let t0 = Instant::now();
    let mut ex = CudaExecutor::new(gpu, &KernelDir::new(kernels), store, layers.as_deref(), cfg)
        .expect("executor with every rung captured");
    let (ladder, capture) = match (ex.decode_graphs(), ex.draft_graphs()) {
        (Some(g), _) => (g.ladder().to_vec(), g.capture_bytes().to_vec()),
        (_, Some(g)) => {
            // Every width's rungs are captured; this bench runs width `depths`.
            let bytes = g
                .rungs()
                .into_iter()
                .zip(g.capture_bytes())
                .filter(|((w, _), _)| *w == depths)
                .map(|(_, &b)| b)
                .collect();
            (g.ladder(depths), bytes)
        }
        _ => unreachable!("graphs on"),
    };
    println!("loaded and captured in {:.1?}", t0.elapsed());
    for (r, bytes) in ladder.iter().zip(&capture) {
        println!("rung {r:>4}: capture took {bytes} bytes of device memory");
    }

    // Every slot's blocks, in every group, mapped once.
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
    let bucket = Bucket {
        max_seqs: max_rows,
        max_tokens,
    };
    // Rows `0..rows` at their positions with their next tokens.
    let decode = |pos: &[u32], tok: &[u32], updates: Vec<TableUpdate>| StepInput {
        bucket,
        maintenance: vec![],
        table_updates: updates,
        seqs: (0u32..)
            .zip(pos)
            .map(|(slot, &p)| SeqEntry {
                slot,
                token_start: slot,
                num_tokens: 1,
                context_len: p,
                num_drafts: depths,
                sample: true,
                sampling: SamplingParams::greedy(),
            })
            .collect(),
        token_ids: tok.to_vec(),
        positions: pos.to_vec(),
        return_logits: false,
    };
    ex.set_decode_path(DecodePath::Eager);
    let start_token = |slot: u32| prompt.as_ref().map_or(1000 + slot, |p| p[p.len() - 1]);
    match &prompt {
        None => {
            ex.execute(&decode(&[context], &[start_token(0)], updates))
                .unwrap();
        }
        Some(p) => {
            // Every row prefilled with the prompt, a block-sized chunk at a
            // time, alone in its step.
            let mut updates = Some(updates);
            for slot in 0..max_rows {
                for start in (0..context).step_by(512) {
                    let len = (context - start).min(512);
                    ex.execute(&StepInput {
                        bucket,
                        maintenance: vec![],
                        table_updates: updates.take().unwrap_or_default(),
                        seqs: vec![SeqEntry {
                            slot,
                            token_start: 0,
                            num_tokens: len,
                            context_len: start,
                            num_drafts: 0,
                            sample: false,
                            sampling: SamplingParams::greedy(),
                        }],
                        token_ids: p[start as usize..(start + len) as usize].to_vec(),
                        positions: (start..start + len).collect(),
                        return_logits: false,
                    })
                    .unwrap();
                }
            }
        }
    }

    println!(
        "\n{:>5} {:>8} {:>11} {:>11} {:>11} {:>12}",
        "rung", "path", "median ms", "p90 ms", "mean ms", "tokens/step"
    );
    let mut json = Vec::new();
    for &rung in &ladder {
        let mut medians = Vec::new();
        for path in [DecodePath::Eager, DecodePath::Direct, DecodePath::Replay] {
            ex.set_decode_path(path);
            let mut times = Vec::with_capacity(iters);
            let mut pos = vec![context; rung as usize];
            let mut tok: Vec<u32> = (0..rung).map(start_token).collect();
            let (mut produced, mut rows_steps) = (0usize, 0usize);
            for i in 0..WARMUP + iters {
                let step = decode(&pos, &tok, vec![]);
                let t = Instant::now();
                let out = ex.execute(&step).unwrap();
                let dt = t.elapsed().as_secs_f64() * 1e3;
                for r in 0..rung as usize {
                    let row = out.row(r);
                    if depths == 0 {
                        // Fixed ids at the next position: every path rewrites
                        // the same KV.
                        pos[r] += 1;
                    } else {
                        pos[r] += u32::try_from(row.len()).unwrap();
                        tok[r] = *row.last().unwrap();
                    }
                    if i >= WARMUP {
                        produced += row.len();
                        rows_steps += 1;
                    }
                }
                if i >= WARMUP {
                    times.push(dt);
                }
            }
            times.sort_by(f64::total_cmp);
            let median = times[times.len() / 2];
            let p90 = times[(times.len() * 9 / 10).min(times.len() - 1)];
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let per_step = produced as f64 / rows_steps as f64;
            println!(
                "{rung:>5} {path:>8?} {median:>11.3} {p90:>11.3} {mean:>11.3} {per_step:>12.3}"
            );
            json.push(format!(
                r#"{{"rung":{rung},"path":"{path:?}","draft_tokens":{depths},"median_ms":{median:.4},"p90_ms":{p90:.4},"mean_ms":{mean:.4},"tokens_per_step":{per_step:.4}}}"#
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
