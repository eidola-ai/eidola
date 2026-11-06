//! Serving a prefix-cache hit from the host tier against recomputing the prefix, and the
//! host tier's copy throughput, on a MiMo-V2.6 checkpoint through the serving core's
//! engine on one executor.
//!
//! ```text
//! host_tier_bench <kernels_dir> <model_dir> [--layers all|<i,j,...>]
//!                 [--prefix 1024,4096,16384,65536] [--iters N] [--chunk N]
//! ```
//!
//! The device's global pool holds `L / block_size + 8` blocks for the longest prefix `L`:
//! one sequence of the displacing request's length (below) and a few blocks more, never two
//! prefixes. So whatever the prefix length, the displacing request cannot fit beside A's
//! cached blocks and must demote them, and A's hit must demote the displacer's. For each
//! prefix length
//! `P` (rounded up to whole blocks) and each of `--iters` runs (default 5), under fresh
//! keys each run:
//!
//! - **cold**: lineage A's request of `P + 1` prompt tokens and one output token,
//!   prefilled in chunks of `--chunk` (default 2048), then flushed (its blocks encrypted);
//! - **displaced**: lineage B's request of `L + 4 × block_size` other tokens (the same at
//!   every prefix length, so its time does not follow `P`), which demotes A's blocks to the
//!   host tier to make room;
//! - **host hit**: A's request again: `P` positions hit, promoted back to the device (which
//!   demotes B in turn), decrypted, and one position computed;
//! - **device hit**: A's request once more, its blocks now on the device: only decrypted;
//! - **miss**: A's prompt under a third key, recomputing all `P + 1` positions (what a hit
//!   saves).
//!
//! Each time is wall time around the engine calls; every step ends in the executor's
//! read-back, so the device work (copies included) is in it. After each run the cache is
//! expired and zeroed (not timed). Then, for raw copy throughput, a store of the same
//! geometry demotes and promotes `P / block_size` blocks of every group in one step each.
//! Prints a table of medians and one JSON line per prefix (`{"prefix", "cold_ms",
//! "displaced_ms", "host_hit_ms", "device_hit_ms", "miss_ms", "promoted_blocks",
//! "image_bytes", "demote_gbps", "promote_gbps"}`).
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::executor::Maintenance;
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::{CacheKey, SaltDeriver};
use eidola_engine::spec::Bucket;
use eidola_engine_cuda::kv::{GroupGeometry, KvStore};
use eidola_engine_cuda::{
    CudaExecutor, CudaExecutorConfig, CudaGraphs, Gpu, KernelDir, KvBlocks, MtpHidden,
};
use eidola_engine_model::safetensors::WeightSet;

const BS: u32 = 16;
const SAMPLEABLE: u32 = 151_675;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|x| x == name)?;
    Some(
        args.get(i + 1)
            .unwrap_or_else(|| panic!("{name} takes a value")),
    )
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Steps until every request has finished, then flushes; milliseconds.
fn drain(eng: &mut Engine<CudaExecutor>, now: &mut u64) -> f64 {
    let t = Instant::now();
    while eng.unfinished() > 0 {
        eng.step(*now).expect("step");
        *now += 1;
    }
    eng.flush().expect("flush");
    t.elapsed().as_secs_f64() * 1e3
}

fn tokens(seed: u64, n: u32) -> Vec<u32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ seed;
    (0..n)
        .map(|_| {
            state = eidola_engine::sampling::mix64(state);
            u32::try_from(state % u64::from(SAMPLEABLE)).unwrap()
        })
        .collect()
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
    let prefixes: Vec<u32> = flag(rest, "--prefix")
        .unwrap_or("1024,4096,16384,65536")
        .split(',')
        .map(|x| x.parse::<u32>().unwrap().next_multiple_of(BS))
        .collect();
    let iters: usize = flag(rest, "--iters").map_or(5, |x| x.parse().unwrap());
    let chunk: u32 = flag(rest, "--chunk").map_or(2048, |x| x.parse().unwrap());
    assert!(iters >= 1 && chunk >= 1 && !prefixes.is_empty());

    let longest = *prefixes.iter().max().unwrap();
    // The displacer: four blocks past the longest prefix, so with A's `P >= 3` blocks
    // cached it cannot fit (its `L / BS + 5` blocks against `L / BS + 7 - P / BS` free), and
    // its own cached blocks leave A's hit only three free blocks.
    let displacer = longest + 4 * BS;
    let max_model_len = (displacer + 2).next_multiple_of(BS) + BS;
    // One sequence of the displacer's length and a few blocks more: never two prefixes.
    let global = longest / BS + 8;
    assert!(
        prefixes.iter().all(|&p| p / BS >= 3),
        "every prefix spans three blocks"
    );
    let host = 4 * (longest / BS + 2) + 8;
    let cfg = CudaExecutorConfig {
        block_size: BS,
        num_blocks: KvBlocks {
            global,
            sliding: 512,
            drafter: 0,
        },
        host_blocks: KvBlocks {
            global: host,
            sliding: 2048,
            drafter: 0,
        },
        num_state_slots: 4,
        max_model_len,
        buckets: vec![
            Bucket {
                max_seqs: 1,
                max_tokens: 16,
            },
            Bucket {
                max_seqs: 4,
                max_tokens: chunk.max(16),
            },
        ],
        sampleable_vocab_size: SAMPLEABLE,
        image: None,
        graphs: CudaGraphs::Off,
        draft_tokens: 0,
        mtp_hidden: MtpHidden::Normed,
    };
    let store = Arc::new(WeightSet::open_dir(Path::new(model)).unwrap());
    let gpu = Gpu::open(0).unwrap();
    let t0 = Instant::now();
    let ex = CudaExecutor::new(gpu, &KernelDir::new(kernels), store, layers.as_deref(), cfg)
        .expect("executor");
    println!("loaded in {:.1?}", t0.elapsed());
    let geometry = ex.kv().geometry().to_vec();
    let sched = SchedulerConfig {
        max_batched_tokens: chunk.max(16),
        max_seqs: 4,
        max_prefill_chunk: chunk,
        eos_token_ids: Vec::new(),
        speculative: false,
        cache: CachePolicy::default(),
        sweep_interval_ms: u64::MAX,
    };
    let max_age = sched.cache.max_age_ms;
    let mut eng = Engine::new(ex, sched).expect("engine");
    let deriver = SaltDeriver::new();
    let mut now = 0u64;
    let mut key_seed = 0u8;
    let mut lineage = || {
        key_seed = key_seed.wrapping_add(1);
        deriver.lineage(&CacheKey::from_bytes([key_seed; 32]))
    };
    let mut id = 0u64;
    let mut request = |prompt: &[u32], cache: CacheScope| {
        id += 1;
        Request {
            id,
            prompt: prompt.to_vec(),
            sampling: SamplingParams::greedy(),
            max_tokens: 1,
            stop_token_ids: Vec::new(),
            cache,
        }
    };

    println!(
        "{:>8} {:>9} {:>12} {:>11} {:>13} {:>9} {:>11} {:>12}",
        "prefix",
        "cold ms",
        "displaced ms",
        "host hit ms",
        "device hit ms",
        "miss ms",
        "demote GB/s",
        "promote GB/s"
    );
    for &p in &prefixes {
        let prompt = tokens(u64::from(p), p + 1);
        let other = tokens(u64::from(p) ^ 0xb0b, displacer + 1);
        let (mut cold, mut displaced, mut host_hit, mut device_hit, mut miss) =
            (vec![], vec![], vec![], vec![], vec![]);
        let mut promoted = 0;
        for _ in 0..iters {
            let a = lineage();
            eng.submit(request(&prompt, CacheScope::Keyed(a.clone())))
                .unwrap();
            cold.push(drain(&mut eng, &mut now));
            eng.submit(request(&other, CacheScope::Keyed(lineage())))
                .unwrap();
            displaced.push(drain(&mut eng, &mut now));
            let before = eng.kv().tier_stats().promoted_blocks;
            eng.submit(request(&prompt, CacheScope::Keyed(a.clone())))
                .unwrap();
            host_hit.push(drain(&mut eng, &mut now));
            promoted = eng.kv().tier_stats().promoted_blocks - before;
            assert!(promoted > 0, "the hit was served from the host tier");
            eng.submit(request(&prompt, CacheScope::Keyed(a))).unwrap();
            device_hit.push(drain(&mut eng, &mut now));
            eng.submit(request(&prompt, CacheScope::Keyed(lineage())))
                .unwrap();
            miss.push(drain(&mut eng, &mut now));
            now += 10 * max_age;
            eng.sweep(now).unwrap();
            assert_eq!(eng.kv_keys_held(), 0);
        }

        // Raw copies: every group's first `P / BS` blocks out and back in one step each.
        let gpu = eng.executor().gpu();
        let blocks: Vec<u32> = geometry
            .iter()
            .map(|g| (p / BS).min(g.num_blocks - 1))
            .collect();
        // A store of just those blocks, so the copies run beside the executor's pools.
        let small = geometry
            .iter()
            .zip(&blocks)
            .map(|(g, &n)| GroupGeometry {
                num_blocks: n + 1,
                ..g.clone()
            })
            .collect();
        let mut raw = KvStore::new(gpu, small, 1, 1, 0).unwrap();
        raw.set_host_tier(gpu, &blocks.iter().map(|&n| n + 1).collect::<Vec<_>>())
            .unwrap();
        let image: usize = geometry
            .iter()
            .zip(&blocks)
            .map(|(g, &n)| g.block_bytes() * n as usize)
            .sum();
        let ops = |demote: bool| -> Vec<Maintenance> {
            let mut m = Vec::new();
            for (g, &n) in blocks.iter().enumerate() {
                let group = u32::try_from(g).unwrap();
                for b in 1..=n {
                    m.push(if demote {
                        Maintenance::Demote {
                            group,
                            block: b,
                            host: b,
                        }
                    } else {
                        Maintenance::Promote {
                            group,
                            host: b,
                            block: b,
                        }
                    });
                }
            }
            m
        };
        let (mut down, mut up) = (vec![], vec![]);
        for _ in 0..iters {
            let t = Instant::now();
            raw.apply(gpu, &ops(true), &[]).unwrap();
            gpu.synchronize().unwrap();
            down.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            raw.apply(gpu, &ops(false), &[]).unwrap();
            gpu.synchronize().unwrap();
            up.push(t.elapsed().as_secs_f64());
        }
        drop(raw);
        let (dg, pg) = (
            image as f64 / median(down) / 1e9,
            image as f64 / median(up) / 1e9,
        );
        let (c, d, h, dh, m) = (
            median(cold),
            median(displaced),
            median(host_hit),
            median(device_hit),
            median(miss),
        );
        println!("{p:>8} {c:>9.2} {d:>12.2} {h:>11.2} {dh:>13.2} {m:>9.2} {dg:>11.2} {pg:>12.2}");
        println!(
            "{{\"prefix\": {p}, \"cold_ms\": {c:.3}, \"displaced_ms\": {d:.3}, \"host_hit_ms\": {h:.3}, \"device_hit_ms\": {dh:.3}, \"miss_ms\": {m:.3}, \"promoted_blocks\": {promoted}, \"image_bytes\": {image}, \"demote_gbps\": {dg:.3}, \"promote_gbps\": {pg:.3}}}"
        );
    }
}
