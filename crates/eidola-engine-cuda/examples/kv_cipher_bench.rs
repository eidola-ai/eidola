//! Decrypting a prefix-cache hit against recomputing the prefix, on a MiMo-V2.6
//! checkpoint, through the serving core's engine on one executor.
//!
//! ```text
//! kv_cipher_bench <kernels_dir> <model_dir> [--layers all|<i,j,...>]
//!                 [--prefix 1024,4096,16384] [--iters N] [--chunk N]
//! ```
//!
//! For each prefix length `P` (rounded up to whole blocks) and each of `--iters` runs
//! (default 5), every run under fresh keys:
//!
//! - **cold**: a keyed request of `P + 1` prompt tokens and one output token, prefilled in
//!   chunks of `--chunk` (default 2048);
//! - **encrypt**: the maintenance-only step after it finishes, which encrypts every block
//!   it left cached (`Engine::flush`); its bytes and throughput are printed;
//! - **hit**: the same request again under the same key: `P` positions hit, so its blocks
//!   are decrypted and one position is computed;
//! - **miss**: the same request under another key, which recomputes all `P + 1` positions
//!   (what the hit saves).
//!
//! Each time is wall time around the engine calls; every step ends in the executor's
//! read-back, so the device work is included. After each run the cache is expired and
//! zeroed (not timed). Prints a table of medians and one JSON line per prefix
//! (`{"prefix", "cold_ms", "encrypt_ms", "encrypt_bytes", "hit_ms", "miss_ms"}`).
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::{CacheKey, SaltDeriver};
use eidola_engine::spec::Bucket;
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

/// Steps until every request has finished; milliseconds.
fn drain(eng: &mut Engine<CudaExecutor>, now: &mut u64) -> f64 {
    let t = Instant::now();
    while eng.unfinished() > 0 {
        eng.step(*now).expect("step");
        *now += 1;
    }
    t.elapsed().as_secs_f64() * 1e3
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
        .unwrap_or("1024,4096,16384")
        .split(',')
        .map(|x| x.parse::<u32>().unwrap().next_multiple_of(BS))
        .collect();
    let iters: usize = flag(rest, "--iters").map_or(5, |x| x.parse().unwrap());
    let chunk: u32 = flag(rest, "--chunk").map_or(2048, |x| x.parse().unwrap());
    assert!(iters >= 1 && chunk >= 1 && !prefixes.is_empty());

    let longest = *prefixes.iter().max().unwrap();
    let max_model_len = (longest + 2).next_multiple_of(BS) + BS;
    // Three lineages' prefixes cached at once (cold, then the miss), plus the running one.
    let global = 4 * (longest / BS + 2) + 8;
    let cfg = CudaExecutorConfig {
        block_size: BS,
        host_blocks: eidola_engine_cuda::KvBlocks::default(),
        num_blocks: KvBlocks {
            global,
            sliding: 512,
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
        CacheScope::Keyed(deriver.lineage(&CacheKey::from_bytes([key_seed; 32])))
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
        "{:>8} {:>10} {:>11} {:>12} {:>10} {:>10} {:>9}",
        "prefix", "cold ms", "encrypt ms", "encrypt MiB", "GB/s", "hit ms", "miss ms"
    );
    for &p in &prefixes {
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ u64::from(p);
        let prompt: Vec<u32> = (0..=p)
            .map(|_| {
                state = eidola_engine::sampling::mix64(state);
                u32::try_from(state % u64::from(SAMPLEABLE)).unwrap()
            })
            .collect();
        let (mut cold, mut enc, mut hit, mut miss) = (vec![], vec![], vec![], vec![]);
        let mut bytes = 0usize;
        for _ in 0..iters {
            let key = lineage();
            let again = match &key {
                CacheScope::Keyed(l) => CacheScope::Keyed(l.clone()),
                CacheScope::Private => unreachable!(),
            };
            eng.submit(request(&prompt, key)).unwrap();
            cold.push(drain(&mut eng, &mut now));
            let geometry = eng.executor().kv().geometry().to_vec();
            bytes = eng
                .kv()
                .pending_crypts()
                .iter()
                .filter(|(_, encrypt)| **encrypt)
                .map(|(&(g, _), _)| geometry[g as usize].block_bytes())
                .sum();
            let t = Instant::now();
            eng.flush().unwrap();
            enc.push(t.elapsed().as_secs_f64() * 1e3);
            eng.submit(request(&prompt, again)).unwrap();
            hit.push(drain(&mut eng, &mut now));
            assert!(eng.kv().cache_entries() > 0, "the hit's entries are cached");
            eng.submit(request(&prompt, lineage())).unwrap();
            miss.push(drain(&mut eng, &mut now));
            eng.flush().unwrap();
            now += 10 * max_age;
            eng.sweep(now).unwrap();
            assert_eq!(eng.kv_keys_held(), 0);
        }
        let (c, e, h, m) = (median(cold), median(enc), median(hit), median(miss));
        let mib = bytes as f64 / (1 << 20) as f64;
        println!(
            "{p:>8} {c:>10.2} {e:>11.3} {mib:>12.1} {:>10.1} {h:>10.2} {m:>9.2}",
            bytes as f64 / (e * 1e6)
        );
        println!(
            "{{\"prefix\": {p}, \"cold_ms\": {c:.3}, \"encrypt_ms\": {e:.3}, \"encrypt_bytes\": {bytes}, \"hit_ms\": {h:.3}, \"miss_ms\": {m:.3}}}"
        );
    }
}
