//! The step streams the engine hands the MiMo- and DeepSeek-shaped mocks, pinned.
//!
//! Each workload (chunked and whole prefill, preemption under small pools, branch,
//! regeneration and continuation hits, required prefixes, the host tier, drafting,
//! expiry and cancellation) runs through a recording executor that hashes a canonical
//! rendering of every `StepInput` it is given and every output it returns, and every
//! event the engine emits. The rendering names each field explicitly, so a field added to
//! the seam leaves the digest alone unless it carries something; a change to what these
//! models are asked to do, in any step, changes the digest.
//!
//! Every request is keyed under a fixed boot key, so salts, nonces and keys are the same
//! on every run; each workload is run twice and must agree with itself before it is
//! compared with the pinned digest.

mod common;

use std::fmt::Write as _;

use common::*;
use eidola_engine::engine::{CacheScope, Engine, Event, Request, SchedulerConfig};
use eidola_engine::executor::{Executor, ExecutorError, Maintenance, StepInput, StepOutput};
use eidola_engine::kv::CachePolicy;
use eidola_engine::mock::{MockConfig, MockExecutor, deepseek_like_spec, with_host_tier};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::ModelSpec;
use sha2::{Digest, Sha256};

/// Hashes everything a step carries in and out.
struct Recorder {
    inner: MockExecutor,
    hash: Sha256,
}

fn render_step(step: &StepInput) -> String {
    assert!(
        step.aux.state_rows.is_empty() && step.aux.state_writes.is_empty(),
        "state rows for a model without state"
    );
    let mut s = String::new();
    let b = step.bucket;
    write!(s, "bucket {} {};", b.max_seqs, b.max_tokens).unwrap();
    for m in &step.maintenance {
        match *m {
            Maintenance::Copy { group, src, dst } => write!(s, "copy {group} {src} {dst};"),
            Maintenance::Zero { group, block } => write!(s, "zero {group} {block};"),
            Maintenance::ResetSlot { slot } => write!(s, "reset {slot};"),
            Maintenance::Encrypt {
                group,
                block,
                key,
                nonce,
            } => write!(s, "enc {group} {block} {key} {:?};", nonce.words()),
            Maintenance::Decrypt {
                group,
                block,
                key,
                nonce,
            } => write!(s, "dec {group} {block} {key} {:?};", nonce.words()),
            Maintenance::Demote { group, block, host } => {
                write!(s, "demote {group} {block} {host};")
            }
            Maintenance::Promote { group, host, block } => {
                write!(s, "promote {group} {host} {block};")
            }
            Maintenance::ZeroHost { group, host } => write!(s, "zerohost {group} {host};"),
            Maintenance::SaveState { .. } => {
                panic!("a state saved for a model without state: {m:?}")
            }
        }
        .unwrap();
    }
    for k in &step.kv_keys {
        let mut w = [0u32; 8];
        k.words_into(&mut w);
        write!(s, "key {w:?};").unwrap();
    }
    for u in &step.table_updates {
        write!(s, "table {} {} {} {};", u.slot, u.group, u.index, u.block).unwrap();
    }
    for e in &step.seqs {
        write!(
            s,
            "row {} {} {} {} {} {} {:?};",
            e.slot, e.token_start, e.num_tokens, e.context_len, e.num_drafts, e.sample, e.sampling
        )
        .unwrap();
    }
    write!(
        s,
        "tokens {:?}; positions {:?}; logits {}; chain {:?}; history {:?};",
        step.token_ids,
        step.positions,
        step.return_logits,
        step.aux.chain_start,
        step.aux.history_ids
    )
    .unwrap();
    s
}

impl Executor for Recorder {
    fn spec(&self) -> &ModelSpec {
        self.inner.spec()
    }

    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
        self.hash.update(render_step(step).as_bytes());
        let out = self.inner.execute(step)?;
        let line = format!(
            "out {:?} {} {:?} {:?}\n",
            out.tokens, out.stride, out.num_tokens, out.logits
        );
        self.hash.update(line.as_bytes());
        Ok(out)
    }
}

fn render_events(hash: &mut Sha256, events: &[Event]) {
    for e in events {
        let line = format!(
            "event {} {:?} {:?} {}\n",
            e.id, e.tokens, e.finish, e.cached_prompt_tokens
        );
        hash.update(line.as_bytes());
    }
}

fn params(i: u64) -> SamplingParams {
    match i % 3 {
        0 => SamplingParams::greedy(),
        1 => SamplingParams::random(1.0, 1000 + i).unwrap(),
        _ => SamplingParams::new(0.7, 8, 0.9, 0.05, 2000 + i).unwrap(),
    }
}

fn keyed(n: u64) -> CacheScope {
    CacheScope::Keyed(salt(n as u8))
}

struct Run {
    eng: Engine<Recorder>,
    events: Sha256,
    now: u64,
    outputs: std::collections::HashMap<u64, Vec<u32>>,
    /// Requests that finished with part of their prompt served from the cache.
    hits: u64,
}

impl Run {
    fn step(&mut self) {
        let events = self.eng.step(self.now).expect("step");
        render_events(&mut self.events, &events);
        for e in &events {
            self.outputs.entry(e.id).or_default().extend(&e.tokens);
            if e.finish.is_some() && e.cached_prompt_tokens > 0 {
                self.hits += 1;
            }
        }
        self.now += 1;
    }

    fn drain(&mut self) {
        let mut guard = 0;
        while self.eng.unfinished() > 0 {
            self.step();
            guard += 1;
            assert!(guard < 100_000, "no progress");
        }
        self.eng.flush().expect("flush");
    }

    fn submit(&mut self, req: Request) {
        self.eng.submit(req).expect("submit");
    }
}

/// One workload: two waves of keyed requests (shared prefixes across lineages, sampled and
/// greedy), then continuations, regenerations and divergent turns of the first wave with
/// required prefixes, a cancellation mid-run, and a wave past the idle TTL.
fn workload(spec: ModelSpec, sched: SchedulerConfig, seed: u64) -> (String, Coverage) {
    let mut run = Run {
        eng: Engine::new(
            Recorder {
                inner: MockExecutor::new(spec.clone(), MockConfig::default()),
                hash: Sha256::new(),
            },
            sched,
        )
        .expect("valid configuration"),
        events: Sha256::new(),
        now: 0,
        outputs: Default::default(),
        hits: 0,
    };
    let vocab = spec.sampleable_vocab_size;
    let mut rng = TestRng(seed);
    let shared = rng.tokens(23, vocab);
    let mut first = Vec::new();
    for id in 0..12u64 {
        let mut prompt = if id % 3 == 0 {
            shared.clone()
        } else {
            Vec::new()
        };
        prompt.extend(rng.var_tokens(1, 40, vocab));
        let max = 1 + rng.below(24) as u32;
        first.push((id, prompt.clone()));
        run.submit(request(id, prompt, params(id), max, keyed(id % 4)));
        if id % 4 == 3 {
            run.step();
        }
    }
    run.drain();

    // Continuations (the prompt, its output and more), regenerations and divergent turns.
    let mut next = 100;
    for (id, prompt) in &first {
        let mut follow = prompt.clone();
        match id % 3 {
            0 => {
                follow.extend(&run.outputs[id]);
                let n = 1 + rng.below(9) as usize;
                follow.extend(rng.tokens(n, vocab));
            }
            1 => {}
            _ => {
                let keep = 1 + rng.below(prompt.len() as u64) as usize;
                follow.truncate(keep);
                follow.extend(rng.tokens(5, vocab));
            }
        }
        let mut req = request(next, follow.clone(), params(next), 12, keyed(id % 4));
        if id % 3 == 1 {
            req.required_cached_tokens = ((prompt.len() as u32 - 1) / 8) * 8;
        }
        run.submit(req);
        next += 1;
    }
    for _ in 0..3 {
        run.step();
    }
    run.eng.cancel(101, run.now);
    run.drain();

    // Past the idle TTL: the sweep scrubs, and a third wave starts cold.
    run.now += 10_000;
    run.eng.sweep(run.now).expect("sweep");
    for (id, prompt) in first.iter().take(4) {
        run.submit(request(
            next,
            prompt.clone(),
            params(next),
            8,
            keyed(id % 4),
        ));
        next += 1;
    }
    run.drain();

    let mut hash = Sha256::new();
    hash.update(run.eng.executor().hash.clone().finalize());
    hash.update(run.events.finalize());
    let mut out = String::new();
    for b in hash.finalize().iter().take(16) {
        write!(out, "{b:02x}").unwrap();
    }
    let stats = run.eng.stats();
    let tier = run.eng.kv().tier_stats();
    let coverage = Coverage {
        preemptions: stats.preemptions,
        drafted: stats.drafted,
        refusals: stats.prefix_refusals,
        demoted: tier.demoted_blocks,
        promoted: tier.promoted_blocks,
        hit: run.hits,
    };
    (out, coverage)
}

/// What a workload exercised, so a pinned digest is known to cover it.
#[derive(Debug, PartialEq, Eq)]
struct Coverage {
    preemptions: u64,
    drafted: u64,
    refusals: u64,
    demoted: u64,
    promoted: u64,
    hit: u64,
}

fn config(chunk: u32, budget: u32, speculative: bool) -> SchedulerConfig {
    SchedulerConfig {
        max_batched_tokens: budget,
        max_seqs: 32,
        max_prefill_chunk: chunk,
        eos_token_ids: vec![7],
        speculative,
        cache: CachePolicy {
            enabled: true,
            idle_ttl_ms: 5_000,
            max_age_ms: 1_000_000,
        },
        sweep_interval_ms: 50,
    }
}

fn mimo(blocks: u32) -> ModelSpec {
    small_spec(blocks)
}

fn deepseek(blocks: u32) -> ModelSpec {
    deepseek_like_spec(32, 4, 2, 3, 3, blocks, 32, 5)
}

/// Runs a workload twice (it must agree with itself) and returns its digest.
fn digest(spec: ModelSpec, sched: SchedulerConfig, seed: u64) -> (String, Coverage) {
    let a = workload(spec.clone(), sched.clone(), seed);
    let b = workload(spec, sched, seed);
    assert_eq!(a, b, "a workload's transcript differs between two runs");
    a
}

#[test]
fn mimo_step_streams_are_pinned() {
    let cases: [(&str, ModelSpec, SchedulerConfig, &str); 5] = [
        (
            "roomy whole",
            mimo(512),
            config(256, 256, false),
            "f52087fe0c1e5f6071c21ebae81e41fc",
        ),
        (
            "roomy drafted",
            mimo(512),
            config(256, 256, true),
            "b5c15266a4c70692036150a5d19581a6",
        ),
        (
            "chunked drafted",
            mimo(512),
            config(5, 64, true),
            "cf4148709e2beeab4eca78083b7c6e16",
        ),
        (
            "tight drafted",
            mimo(48),
            config(7, 64, true),
            "7d290493975fbd843fd0147f47f395d8",
        ),
        (
            "tiered drafted",
            with_host_tier(mimo(40), 96),
            config(11, 64, true),
            "5c4dc76bb748c75095203281983a6a05",
        ),
    ];
    check(&cases, 11);
}

#[test]
fn deepseek_step_streams_are_pinned() {
    let cases: [(&str, ModelSpec, SchedulerConfig, &str); 4] = [
        (
            "roomy whole",
            deepseek(512),
            config(256, 256, false),
            "b8e45ef453c1a408a35818e006ff4a12",
        ),
        (
            "chunked drafted",
            deepseek(512),
            config(5, 64, true),
            "69520a670637b1fe20f5ca02aa5f6ca2",
        ),
        (
            "tight drafted",
            deepseek(64),
            config(7, 64, true),
            "f86b7a5b5ade79e04986501eac637001",
        ),
        (
            "tiered drafted",
            with_host_tier(deepseek(56), 128),
            config(11, 64, true),
            "3a87190cffa7b56a86dec58135c5de42",
        ),
    ];
    check(&cases, 23);
}

fn check(cases: &[(&str, ModelSpec, SchedulerConfig, &str)], seed: u64) {
    let mut wrong = Vec::new();
    for (name, spec, sched, want) in cases {
        let (got, coverage) = digest(spec.clone(), sched.clone(), seed);
        eprintln!("{name}: {coverage:?}");
        assert!(coverage.hit > 0, "{name}: no prefix hit");
        if sched.speculative {
            assert!(coverage.drafted > 0, "{name}: nothing drafted");
        }
        if name.starts_with("tight") {
            assert!(coverage.preemptions > 0, "{name}: nothing preempted");
        }
        if name.starts_with("tiered") {
            assert!(
                coverage.demoted > 0 && coverage.promoted > 0,
                "{name}: the host tier was not used both ways"
            );
        }
        if got != *want {
            wrong.push(format!("{name}: got {got}, pinned {want}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "step streams changed:\n{}",
        wrong.join("\n")
    );
}
