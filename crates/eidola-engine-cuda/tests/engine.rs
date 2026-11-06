//! The serving core's engine over the CUDA executor on the truncated real
//! checkpoint (see `real_flash.rs` for the environment): the CPU executor's
//! workloads (chunked prefill, prefix hits and salt isolation, preemption and
//! resume, cancellation, a randomized mix) adapted to GPU tolerance, and a
//! request's tokens unchanged by prefix hits, chunk sizes and concurrent
//! requests (the executor is batch-invariant).
//!
//! After every step the KV manager's invariants hold and every free block is
//! zero on the device or has a zero queued. Every finished greedy output is
//! checked against the f32 reference forward over the same tokens: each
//! produced token must be the reference's choice or within the measured
//! logit tolerance of it ([`MARGIN`]); sampled outputs must be sampleable ids.
//! With `EIDOLA_ENGINE_CUDA_GRAPHS=on` every workload runs with decode graphs
//! replaying.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use common::setup;
use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::{BootKey, CacheKey, Lineage, SaltDeriver};
use eidola_engine::spec::Bucket;
use eidola_engine_cuda::{CudaExecutor, CudaExecutorConfig, CudaGraphs, KvBlocks, MtpHidden};
use eidola_engine_model::safetensors::WeightSet;
use eidola_engine_model::{ForwardOptions, LoadOptions, LogitsAt, ModelWeights, ReferenceModel};

const KEEP: [usize; 4] = [0, 1, 2, 5];
const SAMPLEABLE: u32 = 151_675;
/// A produced greedy token may trail the reference's argmax by at most this
/// many logits: the GPU's measured max |Δlogit| against the reference on this
/// truncation is 1.18 (quantization-explained; see the crate AGENTS.md).
const MARGIN: f32 = 1.5;

struct Env {
    store: Arc<WeightSet>,
    reference: Arc<ReferenceModel>,
    text: Vec<u32>,
}

fn env() -> Option<&'static Env> {
    static ENV: OnceLock<Option<Env>> = OnceLock::new();
    ENV.get_or_init(|| {
        let (Some(dir), Some(golden)) = (
            std::env::var_os("EIDOLA_MIMO_DIR"),
            std::env::var_os("EIDOLA_MIMO_GOLDEN"),
        ) else {
            eprintln!("skipping: EIDOLA_MIMO_DIR / EIDOLA_MIMO_GOLDEN not set");
            return None;
        };
        let g = WeightSet::open_files(&[PathBuf::from(golden)]).unwrap();
        let text = g
            .get("tokens")
            .unwrap()
            .to_i64("tokens")
            .unwrap()
            .into_iter()
            .map(|x| u32::try_from(x).unwrap())
            .collect();
        let store = Arc::new(WeightSet::open_dir(&PathBuf::from(dir)).unwrap());
        let config = store.model_config().unwrap().truncated(&KEEP).unwrap();
        let opts = LoadOptions {
            load_mtp: false,
            ..LoadOptions::default()
        };
        let reference = Arc::new(ReferenceModel::new(
            ModelWeights::load(store.clone(), config, &opts).unwrap(),
        ));
        Some(Env {
            store,
            reference,
            text,
        })
    })
    .as_ref()
}

/// `EIDOLA_ENGINE_CUDA_GRAPHS` (`on` / `off`, off when unset): the whole
/// suite runs with decode graphs replaying when it is `on`.
fn graphs_from_env() -> CudaGraphs {
    std::env::var("EIDOLA_ENGINE_CUDA_GRAPHS").map_or(CudaGraphs::Off, |v| {
        CudaGraphs::parse(&v).unwrap_or_else(|e| panic!("EIDOLA_ENGINE_CUDA_GRAPHS: {e}"))
    })
}

fn sched(chunk: u32) -> SchedulerConfig {
    SchedulerConfig {
        max_batched_tokens: 256,
        max_seqs: 8,
        max_prefill_chunk: chunk,
        eos_token_ids: Vec::new(),
        speculative: false,
        cache: CachePolicy::default(),
        sweep_interval_ms: 1000,
    }
}

/// The lineage of client key `[n; 32]` under a fixed boot key.
fn salt(n: u8) -> Lineage {
    SaltDeriver::with_boot_key(BootKey::from_bytes([0; 32])).lineage(&CacheKey::from_bytes([n; 32]))
}

fn request(
    id: u64,
    prompt: Vec<u32>,
    sampling: SamplingParams,
    max: u32,
    cache: CacheScope,
) -> Request {
    Request {
        id,
        prompt,
        sampling,
        max_tokens: max,
        stop_token_ids: Vec::new(),
        cache,
    }
}

struct Harness {
    eng: Engine<CudaExecutor>,
    now: u64,
    outputs: HashMap<u64, Vec<u32>>,
    finished: HashMap<u64, FinishReason>,
    cached: HashMap<u64, u32>,
    prompts: HashMap<u64, (Vec<u32>, SamplingParams)>,
    zero_checks: usize,
}

impl Harness {
    fn new(blocks: u32, chunk: u32) -> Option<Harness> {
        Harness::with_len(blocks, chunk, 1024)
    }

    fn with_len(blocks: u32, chunk: u32, max_model_len: u32) -> Option<Harness> {
        Harness::with_host(blocks, 0, chunk, max_model_len)
    }

    /// With `host` host-tier blocks per group (0: none).
    fn with_host(blocks: u32, host: u32, chunk: u32, max_model_len: u32) -> Option<Harness> {
        let env = env()?;
        let su = setup()?;
        let cfg = CudaExecutorConfig {
            block_size: 16,
            host_blocks: KvBlocks {
                global: host,
                sliding: host,
                drafter: 0,
            },
            num_blocks: KvBlocks {
                global: blocks,
                sliding: blocks,
                drafter: 0,
            },
            num_state_slots: 8,
            max_model_len,
            buckets: vec![
                Bucket {
                    max_seqs: 1,
                    max_tokens: 16,
                },
                Bucket {
                    max_seqs: 8,
                    max_tokens: 256,
                },
            ],
            sampleable_vocab_size: SAMPLEABLE,
            image: None,
            graphs: graphs_from_env(),
            draft_tokens: 0,
            mtp_hidden: MtpHidden::Normed,
        };
        let ex = CudaExecutor::new(su.gpu, &su.dir, env.store.clone(), Some(&KEEP), cfg).unwrap();
        Some(Harness {
            eng: Engine::new(ex, sched(chunk)).expect("valid configuration"),
            now: 0,
            outputs: HashMap::new(),
            finished: HashMap::new(),
            cached: HashMap::new(),
            prompts: HashMap::new(),
            zero_checks: 0,
        })
    }

    fn submit(&mut self, req: Request) {
        self.outputs.insert(req.id, Vec::new());
        self.prompts
            .insert(req.id, (req.prompt.clone(), req.sampling));
        self.eng.submit(req).expect("submit");
    }

    fn check(&mut self) {
        let kv = self.eng.kv();
        kv.check_invariants();
        kv.check_no_reservations();
        let pending = kv.pending_zeros();
        for (g, b) in kv.free_block_set() {
            if !pending.contains(&(g, b)) {
                assert!(
                    self.eng.executor().block_is_zero(g, b).unwrap(),
                    "free block {g}/{b} holds data with no zero pending"
                );
                self.zero_checks += 1;
            }
        }
        let host_zeros = kv.pending_host_zeros();
        for (g, hb) in kv.free_host_set() {
            if !host_zeros.contains(&(g, hb)) {
                assert!(
                    self.eng.executor().host_block_is_zero(g, hb).unwrap(),
                    "free host block {g}/{hb} holds data with no zero pending"
                );
                self.zero_checks += 1;
            }
        }
    }

    fn step(&mut self) {
        let events = self.eng.step(self.now).expect("step");
        for e in events {
            self.outputs.entry(e.id).or_default().extend(&e.tokens);
            if let Some(f) = e.finish {
                assert!(self.finished.insert(e.id, f).is_none(), "finished twice");
                self.cached.insert(e.id, e.cached_prompt_tokens);
            }
        }
        self.check();
    }

    fn run(&mut self) {
        let mut guard = 0;
        while self.eng.unfinished() > 0 {
            self.step();
            self.now += 1;
            guard += 1;
            assert!(guard < 100_000, "no progress");
        }
    }

    /// Greedy outputs against the reference within [`MARGIN`]; every output
    /// sampleable. Returns (greedy tokens checked, of which the reference's
    /// own argmax, worst trailing margin).
    fn check_outputs(&self) -> (usize, usize, f32) {
        let env = env().unwrap();
        let (mut n, mut exact, mut worst) = (0, 0, 0f32);
        let mut ids: Vec<_> = self.finished.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            let out = &self.outputs[&id];
            assert!(
                out.iter().all(|&t| t < SAMPLEABLE),
                "request {id}: padded id"
            );
            let (prompt, params) = &self.prompts[&id];
            if !params.is_greedy() || out.is_empty() {
                continue;
            }
            let mut tokens = prompt.clone();
            tokens.extend(&out[..out.len() - 1]);
            let positions: Vec<usize> = (prompt.len() - 1..tokens.len()).collect();
            let r = env
                .reference
                .forward(
                    &tokens,
                    &ForwardOptions {
                        logits: LogitsAt::Positions(positions),
                        capture_layers: false,
                    },
                )
                .unwrap();
            for (i, &t) in out.iter().enumerate() {
                let row = &r.logits.row(i)[..SAMPLEABLE as usize];
                let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let trail = max - row[t as usize];
                worst = worst.max(trail);
                exact += (trail == 0.0) as usize;
                n += 1;
                assert!(
                    trail <= MARGIN,
                    "request {id} token {i}: {t} trails the reference argmax by {trail}"
                );
            }
        }
        (n, exact, worst)
    }
}

fn greedy() -> SamplingParams {
    SamplingParams::greedy()
}

/// Idle prefix-cache KV is ciphertext on the device, and a hit decrypts it into exactly
/// the bits a recompute writes. A keyed request runs (each block's bytes read back while a
/// sequence maps it, or its encryption is still queued, which is its plaintext); once the
/// engine flushes, every cached block on the device differs from that plaintext in
/// essentially every word, the cipher's device item list (which carried the key) is zero,
/// and the engine holds no key. The same request again hits and produces the same tokens
/// as the first run and as a run under another key, which recomputes everything.
#[test]
fn idle_kv_is_ciphertext_on_the_device_and_a_hit_decrypts_it() {
    let Some(mut h) = Harness::new(64, 64) else {
        return;
    };
    let text = env().unwrap().text.clone();
    let prompt = text[..70].to_vec();
    h.submit(request(
        1,
        prompt.clone(),
        greedy(),
        8,
        CacheScope::Keyed(salt(1)),
    ));
    let mut plain: HashMap<(u32, u32), Vec<u16>> = HashMap::new();
    while h.eng.unfinished() > 0 {
        h.step();
        h.now += 1;
        let kv = h.eng.kv();
        let crypts = kv.pending_crypts();
        let mut blocks: Vec<(u32, u32)> = kv
            .sequence_blocks()
            .into_iter()
            .filter(|b| !crypts.contains_key(b))
            .collect();
        blocks.extend(crypts.iter().filter(|(_, e)| **e).map(|(b, _)| *b));
        let ex = h.eng.executor();
        for (g, b) in blocks {
            plain.insert((g, b), ex.kv().read_block(ex.gpu(), g as usize, b).unwrap());
        }
    }
    h.eng.flush().unwrap();
    h.check();
    let ex = h.eng.executor();
    let cached = h.eng.kv().cache_blocks();
    assert_eq!(h.eng.kv().idle_blocks(), cached);
    for &(g, b) in &cached {
        let c = ex.kv().read_block(ex.gpu(), g as usize, b).unwrap();
        let p = &plain[&(g, b)];
        let same = c
            .as_chunks::<2>()
            .0
            .iter()
            .zip(p.as_chunks::<2>().0)
            .filter(|(x, y)| x == y)
            .count();
        assert!(
            same * 1000 < c.len() / 2,
            "group {g} block {b}: idle KV is not ciphertext ({same} words unchanged)"
        );
    }
    assert!(
        ex.kv()
            .cipher()
            .unwrap()
            .read_items(ex.gpu())
            .unwrap()
            .iter()
            .all(|i| *i == eidola_engine_cuda::kv_cipher::CipherItem::default()),
        "the device item list still holds a key"
    );
    assert_eq!(h.eng.kv_keys_held(), 0);
    h.submit(request(
        2,
        prompt.clone(),
        greedy(),
        8,
        CacheScope::Keyed(salt(1)),
    ));
    h.submit(request(3, prompt, greedy(), 8, CacheScope::Keyed(salt(2))));
    h.run();
    assert_eq!(h.cached[&2], 64, "a hit on the decrypted prefix");
    assert_eq!(h.cached[&3], 0);
    assert_eq!(h.outputs[&2], h.outputs[&1], "the hit changed the output");
    assert_eq!(
        h.outputs[&3], h.outputs[&1],
        "the recompute changed the output"
    );
    h.eng.flush().unwrap();
    assert_eq!(h.eng.kv_keys_held(), 0);
    let (n, _, worst) = h.check_outputs();
    println!("idle encryption: {n} greedy tokens, worst trail {worst:.3}");
}

#[test]
fn chunked_prefill_and_prefix_hits() {
    let Some(mut h) = Harness::new(64, 16) else {
        return;
    };
    let text = env().unwrap().text.clone();
    h.submit(request(
        1,
        text[..40].to_vec(),
        greedy(),
        8,
        CacheScope::Keyed(salt(1)),
    ));
    h.submit(request(
        2,
        text[100..130].to_vec(),
        SamplingParams::new(0.8, 40, 0.9, 0.0, 7).unwrap(),
        8,
        CacheScope::Private,
    ));
    h.run();
    // The same conversation, longer: resumes from the cached prefix.
    h.submit(request(
        3,
        text[..70].to_vec(),
        greedy(),
        8,
        CacheScope::Keyed(salt(1)),
    ));
    // Another salt sees nothing.
    h.submit(request(
        4,
        text[..70].to_vec(),
        greedy(),
        8,
        CacheScope::Keyed(salt(2)),
    ));
    h.run();
    assert_eq!(h.cached[&3], 32, "prefix hit");
    assert_eq!(h.cached[&4], 0, "salt isolation");
    let (n, exact, worst) = h.check_outputs();
    println!(
        "chunked prefill + prefix hits: {n} greedy tokens, {exact} the reference argmax, worst trail {worst:.3}; {} zero checks",
        h.zero_checks
    );
    // Hit or not, the same greedy continuation.
    assert_eq!(
        h.outputs[&3], h.outputs[&4],
        "a prefix hit changed the output"
    );
}

/// A request's tokens, greedy and seeded, are the same whether its prefix
/// was recomputed or hit in the cache, whatever the prefill chunk, and
/// whatever runs beside it: prompts past the first global chunk boundary
/// (1,024 positions), run alone in chunks of 16, alone in chunks of 256,
/// beside three other requests, and after an earlier request under the same
/// key cached their prefix.
#[test]
fn prefix_hits_chunking_and_neighbours_change_no_token() {
    let Some(env) = env() else { return };
    let text: Vec<u32> = env.text.iter().copied().cycle().take(1300).collect();
    let jobs: Vec<(u64, Vec<u32>, SamplingParams)> = vec![
        (1, text[..1100].to_vec(), greedy()),
        (
            2,
            text[37..1090].to_vec(),
            SamplingParams::random(1.0, 9).unwrap(),
        ),
        (
            3,
            text[200..1250].to_vec(),
            SamplingParams::new(0.8, 40, 0.95, 0.02, 11).unwrap(),
        ),
    ];
    let run = |chunk: u32, neighbours: bool, warm: bool| -> Option<HashMap<u64, Vec<u32>>> {
        let mut h = Harness::with_len(400, chunk, 2048)?;
        if warm {
            // The same prompts under the same key, first: their prefixes are
            // cached when the measured requests arrive.
            for (id, prompt, params) in &jobs {
                h.submit(request(
                    100 + id,
                    prompt.clone(),
                    *params,
                    4,
                    CacheScope::Keyed(salt(u8::try_from(*id).unwrap())),
                ));
            }
            h.run();
        }
        for (id, prompt, params) in &jobs {
            h.submit(request(
                *id,
                prompt.clone(),
                *params,
                24,
                CacheScope::Keyed(salt(u8::try_from(*id).unwrap())),
            ));
            if neighbours {
                h.submit(request(
                    10 + id,
                    text[300 + 50 * usize::try_from(*id).unwrap()..][..400].to_vec(),
                    greedy(),
                    24,
                    CacheScope::Private,
                ));
            }
        }
        h.run();
        if warm {
            for (id, _, _) in &jobs {
                assert!(h.cached[id] >= 1024, "request {id}: a prefix hit");
            }
        }
        Some(
            jobs.iter()
                .map(|(id, _, _)| (*id, h.outputs[id].clone()))
                .collect(),
        )
    };
    let Some(base) = run(16, false, false) else {
        return;
    };
    for (chunk, neighbours, warm) in [(256, false, false), (64, true, false), (16, false, true)] {
        let got = run(chunk, neighbours, warm).unwrap();
        for (id, _, _) in &jobs {
            assert_eq!(
                got[id], base[id],
                "request {id}: chunk {chunk}, neighbours {neighbours}, cached prefix {warm}"
            );
        }
        println!(
            "chunk {chunk}, neighbours {neighbours}, cached prefix {warm}: every token the same"
        );
    }
}

#[test]
fn preemption_and_cancellation_under_pressure() {
    // Few blocks: concurrent requests preempt each other and resume.
    let Some(mut h) = Harness::new(14, 32) else {
        return;
    };
    let text = env().unwrap().text.clone();
    for id in 0..6u64 {
        let start = usize::try_from(id * 37).unwrap();
        let params = if id % 2 == 0 {
            greedy()
        } else {
            SamplingParams::random(1.0, 100 + id).unwrap()
        };
        h.submit(request(
            id,
            text[start..start + 40].to_vec(),
            params,
            12,
            CacheScope::Private,
        ));
    }
    // Cancel one mid-flight.
    for _ in 0..4 {
        h.step();
        h.now += 1;
    }
    if let Some(e) = h.eng.cancel(5, h.now) {
        h.finished.insert(5, e.finish.unwrap());
    }
    h.run();
    let stats = h.eng.stats();
    println!("stats: {stats:?}");
    let (n, exact, worst) = h.check_outputs();
    println!(
        "preemption + cancellation: {n} greedy tokens, {exact} the reference argmax, worst trail {worst:.3}; {} zero checks",
        h.zero_checks
    );
    for id in 0..5 {
        assert_eq!(h.finished[&id], FinishReason::Length, "request {id}");
    }
    // Everything released and zeroed (or queued) once idle; one sweep step
    // flushes the queued zeros.
    h.eng.sweep(h.now + 1).unwrap();
    h.check();
}

#[test]
fn randomized_workload() {
    let Some(mut h) = Harness::new(48, 24) else {
        return;
    };
    let text = env().unwrap().text.clone();
    let mut s = 7u64;
    let mut rnd = |n: u64| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) % n
    };
    let mut id = 0u64;
    for round in 0..4 {
        for _ in 0..4 {
            let start = usize::try_from(rnd(200)).unwrap();
            let len = 5 + usize::try_from(rnd(60)).unwrap();
            let cache = if rnd(2) == 0 {
                CacheScope::Keyed(salt(u8::try_from(rnd(3)).unwrap()))
            } else {
                CacheScope::Private
            };
            let params = match rnd(3) {
                0 => greedy(),
                1 => SamplingParams::random(0.7, id).unwrap(),
                _ => SamplingParams::new(1.0, 50, 0.95, 0.02, id).unwrap(),
            };
            h.submit(request(
                id,
                text[start..start + len].to_vec(),
                params,
                1 + u32::try_from(rnd(10)).unwrap(),
                cache,
            ));
            id += 1;
        }
        for _ in 0..3 + round {
            h.step();
            h.now += 1;
        }
    }
    h.run();
    let (n, exact, worst) = h.check_outputs();
    println!(
        "randomized: {} requests, {n} greedy tokens, {exact} the reference argmax, worst trail {worst:.3}; {} zero checks",
        id, h.zero_checks
    );
}

/// Runs request `id` alone to completion and flushes.
fn run_alone(h: &mut Harness, id: u64, prompt: Vec<u32>, key: u8) {
    h.submit(request(
        id,
        prompt,
        greedy(),
        8,
        CacheScope::Keyed(salt(key)),
    ));
    h.run();
    h.eng.flush().unwrap();
    h.check();
}

/// The host tier on the device: a conversation pushed off small device pools
/// by others is demoted (its blocks reach host memory as ciphertext: not the
/// plaintext a sequence read while mapping them), and its continuation is
/// served from promoted, decrypted blocks with the same tokens as a run whose
/// cache never left the device and as a recompute under another key. Every
/// free host block is zero after every step (the harness's check).
#[test]
fn a_host_tier_hit_equals_a_device_hit_and_a_recompute() {
    let text = match env() {
        Some(e) => e.text.clone(),
        None => return,
    };
    // Disjoint spans of the golden: the conversation, five others that push it off the
    // device, and its continuation's new tokens.
    const SPAN: usize = 50;
    assert!(
        text.len() >= 6 * SPAN + 5,
        "the golden holds {} tokens; this test needs {}",
        text.len(),
        6 * SPAN + 5
    );
    let prompt = text[..SPAN].to_vec();
    let others: Vec<Vec<u32>> = (1..=5)
        .map(|i| text[i * SPAN..(i + 1) * SPAN].to_vec())
        .collect();
    let tail = &text[6 * SPAN..6 * SPAN + 5];
    let mut runs = Vec::new();
    // 16 blocks hold the conversation and three others' cached blocks, not six.
    for (blocks, host) in [(256, 0), (16, 128)] {
        let Some(mut h) = Harness::with_host(blocks, host, 64, 1024) else {
            return;
        };
        let mut plain: HashMap<(u32, u32), Vec<u16>> = HashMap::new();
        h.submit(request(
            1,
            prompt.clone(),
            greedy(),
            8,
            CacheScope::Keyed(salt(1)),
        ));
        while h.eng.unfinished() > 0 {
            h.step();
            h.now += 1;
            let kv = h.eng.kv();
            let crypts = kv.pending_crypts();
            let mut blocks: Vec<(u32, u32)> = kv
                .sequence_blocks()
                .into_iter()
                .filter(|b| !crypts.contains_key(b))
                .collect();
            blocks.extend(crypts.iter().filter(|(_, e)| **e).map(|(b, _)| *b));
            let ex = h.eng.executor();
            for (g, b) in blocks {
                plain.insert((g, b), ex.kv().read_block(ex.gpu(), g as usize, b).unwrap());
            }
        }
        h.eng.flush().unwrap();
        h.check();
        let first: Vec<(u32, u32)> = h.eng.kv().cache_blocks().into_iter().collect();
        for (i, p) in others.iter().enumerate() {
            let i = u8::try_from(i).unwrap();
            run_alone(&mut h, 10 + u64::from(i), p.clone(), 10 + i);
        }
        if host > 0 {
            let tier = h.eng.kv().tier_stats();
            assert!(
                tier.demoted_blocks > 0 && tier.device_drops == 0,
                "{tier:?}"
            );
            let ex = h.eng.executor();
            for (g, hb) in h.eng.kv().host_cache_blocks() {
                let img = ex.kv().read_host_block(ex.gpu(), g as usize, hb).unwrap();
                for &(pg, pb) in &first {
                    if pg != g {
                        continue;
                    }
                    let p: Vec<u8> = plain[&(pg, pb)]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect();
                    let same = img.iter().zip(&p).filter(|(a, b)| a == b).count();
                    assert!(same * 64 < img.len(), "group {g}: plaintext on the host");
                }
            }
        }
        let mut next = prompt.clone();
        next.extend(&h.outputs[&1]);
        next.extend(tail);
        run_alone(&mut h, 2, next.clone(), 1);
        if host > 0 {
            assert!(h.eng.kv().tier_stats().promoted_blocks > 0);
        }
        run_alone(&mut h, 3, next, 99);
        assert_eq!(h.cached[&3], 0);
        assert_eq!(
            h.outputs[&3], h.outputs[&2],
            "a recompute gives the same tokens"
        );
        let (n, _, worst) = h.check_outputs();
        println!(
            "host {host}: cached {} tokens; {n} greedy tokens, worst trail {worst:.3}",
            h.cached[&2]
        );
        runs.push((h.outputs[&2].clone(), h.cached[&2]));
    }
    assert_eq!(runs[0], runs[1], "a host-tier hit equals a device hit");
    assert!(runs[0].1 > 0);
}
