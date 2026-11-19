//! Reproducibility between calls, asserted: a request's greedy tokens and every logit it
//! is sampled from are the same bits whatever else shares its steps, however its prompt is
//! chunked, whether a position was decoded or prefilled, whether its prompt came from a
//! cache hit, and (greedy) at every draft width. Run on any device and profile; this is
//! the contract the fast profile serves under.
//!
//! A [`Recorder`] wraps the executor: every step asks for logits, and each produced
//! token's logits are kept under the tokens before it (the slot's history, rebuilt from
//! the steps and, for a slot starting past position 0, from the registered prompts).
//! Two runs then agree wherever both computed a prefix.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use eidola_engine::engine::{CacheScope, Engine};
use eidola_engine::executor::{Executor, ExecutorError, Maintenance, StepInput, StepOutput};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::ModelSpec;
use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath, QwenCudaExecutor, QwenDevice};

use super::pair::{Geometry, TestRng, VOCAB, cuda_config, fixture, keyed, request, sched};

/// Logits bits by the token prefix they follow.
pub type Book = HashMap<Vec<u32>, Vec<u32>>;

pub struct Recorder<E: Executor> {
    pub inner: E,
    pub book: Book,
    prompts: Mutex<Vec<Vec<u32>>>,
    history: HashMap<u32, Vec<u32>>,
}

impl<E: Executor> Recorder<E> {
    fn prefix_for(&self, c: u32, host: &[u32]) -> Option<Vec<u32>> {
        let c = c as usize;
        let prompts = self.prompts.lock().expect("prompts");
        prompts
            .iter()
            .find(|p| p.len() >= c + host.len() && p[c..c + host.len()] == *host)
            .map(|p| p[..c].to_vec())
    }
}

impl<E: Executor> Executor for Recorder<E> {
    fn spec(&self) -> &ModelSpec {
        self.inner.spec()
    }

    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
        for m in &step.maintenance {
            if let Maintenance::ResetSlot { slot } = m {
                self.history.remove(slot);
            }
        }
        let mut s = step.clone();
        s.return_logits = true;
        let out = self.inner.execute(&s)?;
        let logits = out.logits.as_ref().expect("logits");
        for (r, e) in step.seqs.iter().enumerate() {
            let a = e.token_start as usize;
            let host = &step.token_ids[a..a + e.num_tokens as usize];
            let hist = match self.history.get(&e.slot) {
                Some(h) if h.len() == e.context_len as usize => Some(h.clone()),
                _ => self.prefix_for(e.context_len, host),
            };
            let Some(mut hist) = hist else {
                self.history.remove(&e.slot);
                continue;
            };
            hist.extend_from_slice(host);
            let n = out.num_tokens[r] as usize;
            #[allow(clippy::needless_range_loop)]
            for j in 0..n {
                let bits: Vec<u32> = logits[r][j].iter().map(|v| v.to_bits()).collect();
                if let Some(old) = self.book.insert(hist.clone(), bits.clone()) {
                    assert_eq!(old, bits, "one prefix, two logits in one run");
                }
                hist.push(out.tokens[r * out.stride as usize + j]);
            }
            // The last produced token is not computed yet: the history stands at its
            // position.
            if n > 0 {
                hist.pop();
            }
            self.history.insert(e.slot, hist);
        }
        let mut o = out;
        if !step.return_logits {
            o.logits = None;
        }
        Ok(o)
    }
}

pub struct Run<D: QwenDevice> {
    pub eng: Engine<Recorder<QwenCudaExecutor<D>>>,
    pub outputs: HashMap<u64, Vec<u32>>,
    /// Prompt positions each finished request was served from the cache.
    pub cached: HashMap<u64, u32>,
    now: u64,
}

impl<D: QwenDevice> Run<D> {
    pub fn new(dev: D, k: u32, chunk: u32) -> Self {
        Self::with_config(dev, k, chunk, cuda_config(&Geometry::plain(k)))
    }

    /// [`Run::new`] for sequences up to 2,048 positions: past several of FA2's 512-key
    /// chunks, and with prefill chunks whose attention outgrows the partial rows of one
    /// pass.
    pub fn long(dev: D, chunk: u32) -> Self {
        let mut c = cuda_config(&Geometry::plain(0));
        c.max_model_len = 2048;
        Self::with_config(dev, 0, chunk, c)
    }

    fn with_config(
        dev: D,
        k: u32,
        chunk: u32,
        cfg: eidola_engine_cuda::qwen::QwenCudaExecutorConfig,
    ) -> Self {
        let model = fixture();
        let mut ex = QwenCudaExecutor::new(dev, &model, cfg).expect("executor");
        super::apply_path(&mut ex);
        let rec = Recorder {
            inner: ex,
            book: Book::new(),
            prompts: Mutex::new(Vec::new()),
            history: HashMap::new(),
        };
        Run {
            eng: Engine::new(rec, sched(chunk, k > 0)).expect("config"),
            outputs: HashMap::new(),
            cached: HashMap::new(),
            now: 0,
        }
    }

    pub fn submit(
        &mut self,
        id: u64,
        prompt: Vec<u32>,
        p: SamplingParams,
        max: u32,
        cache: CacheScope,
    ) {
        // The recorder rebuilds a hit slot's history from the prompts registered.
        self.eng
            .executor()
            .prompts
            .lock()
            .expect("prompts")
            .push(prompt.clone());
        self.outputs.insert(id, Vec::new());
        self.eng
            .submit(request(id, prompt, p, max, cache))
            .expect("submit");
    }

    pub fn run(&mut self) {
        let mut guard = 0;
        while self.eng.unfinished() > 0 {
            for e in self.eng.step(self.now).expect("step") {
                self.outputs.entry(e.id).or_default().extend(&e.tokens);
                if e.finish.is_some() {
                    self.cached.insert(e.id, e.cached_prompt_tokens);
                }
            }
            self.now += 1;
            guard += 1;
            assert!(guard < 100_000, "no progress");
        }
        self.eng.flush().expect("flush");
        let s = self.eng.executor().inner.decode_stats();
        super::note_rung_steps(s.direct + s.replay);
    }

    pub fn book(&self) -> &Book {
        &self.eng.executor().book
    }
}

/// Every prefix both books hold has the same logits bits; returns how many were compared.
pub fn agree(what: &str, a: &Book, b: &Book) -> Result<usize, String> {
    let mut n = 0;
    for (k, v) in a {
        if let Some(w) = b.get(k) {
            if v != w {
                let i = v.iter().zip(w).position(|(x, y)| x != y).unwrap_or(0);
                return Err(format!(
                    "{what}: logits after {} tokens differ at id {i}: {:#x} vs {:#x}",
                    k.len(),
                    v[i],
                    w[i]
                ));
            }
            n += 1;
        }
    }
    Ok(n)
}

fn ensure(ok: bool, what: String) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what) }
}

/// Every invariance scenario on devices from `make`; one line per scenario.
pub fn all<D: QwenDevice>(make: &dyn Fn() -> D) -> Vec<(&'static str, Result<String, String>)> {
    let g = SamplingParams::greedy();
    let mut rng = TestRng(4242);
    let prompt = rng.tokens(19, VOCAB);
    let crowd: Vec<Vec<u32>> = [3usize, 11, 26, 7, 15]
        .iter()
        .map(|&n| rng.tokens(n, VOCAB))
        .collect();
    let mut out = Vec::new();

    // A request alone, and beside traffic of varied sizes arriving at varied steps.
    let alone = {
        let mut r = Run::new(make(), 0, 256);
        r.submit(1, prompt.clone(), g, 10, keyed(1));
        r.run();
        r
    };
    let res = (|| {
        let mut r = Run::new(make(), 0, 5);
        let mut all = crowd.clone();
        all.push(prompt.clone());
        for (i, c) in crowd.iter().enumerate() {
            let id = 10 + u64::try_from(i).expect("small");
            r.submit(
                id,
                c.clone(),
                super::pair::params(id),
                6 + u32::try_from(i).expect("small"),
                keyed(2),
            );
            if i == 2 {
                r.submit(1, prompt.clone(), g, 10, keyed(1));
            }
            for e in r.eng.step(r.now).expect("step") {
                r.outputs.entry(e.id).or_default().extend(&e.tokens);
            }
            r.now += 1;
        }
        r.run();
        ensure(
            r.outputs[&1] == alone.outputs[&1],
            "greedy tokens differ beside traffic".into(),
        )?;
        let n = agree("beside traffic", alone.book(), r.book())?;
        ensure(n >= 10, format!("only {n} prefixes compared"))?;
        Ok(format!("{n} prefixes"))
    })();
    out.push(("alone_and_beside_traffic", res));

    // Prefilled in one chunk and in several.
    let res = (|| {
        let mut total = 0;
        for chunk in [1u32, 3, 4, 7] {
            let mut r = Run::new(make(), 0, chunk);
            r.submit(1, prompt.clone(), g, 10, keyed(1));
            r.run();
            ensure(
                r.outputs[&1] == alone.outputs[&1],
                format!("tokens differ at chunk {chunk}"),
            )?;
            total += agree(&format!("chunk {chunk}"), alone.book(), r.book())?;
        }
        Ok(format!("{total} prefixes over chunks 1, 3, 4, 7"))
    })();
    out.push(("chunking", res));

    // Decoded positions against the same positions prefilled.
    let res = (|| {
        let mut total = 0;
        let generated = alone.outputs[&1].clone();
        for m in [1usize, 4, 9] {
            let mut p = prompt.clone();
            p.extend_from_slice(&generated[..m]);
            let mut r = Run::new(make(), 0, 6);
            r.submit(1, p, g, 1, CacheScope::Private);
            r.run();
            let n = agree(&format!("prefilled through {m}"), alone.book(), r.book())?;
            ensure(
                n == 1,
                format!("prefill through {m}: {n} prefixes compared"),
            )?;
            total += n;
        }
        Ok(format!("{total} positions"))
    })();
    out.push(("decoded_and_prefilled", res));

    // A second turn resumed from the first turn's snapshot, and computed fresh.
    let res = (|| {
        let mut turn = prompt.clone();
        turn.extend_from_slice(&alone.outputs[&1]);
        turn.extend(TestRng(77).tokens(6, VOCAB));
        let mut hit = Run::new(make(), 0, 256);
        hit.submit(1, prompt.clone(), g, 10, keyed(1));
        hit.run();
        hit.submit(2, turn.clone(), g, 8, keyed(1));
        hit.run();
        let cached = hit.cached[&2];
        ensure(
            cached > 0,
            "the second turn did not hit the first's snapshot".into(),
        )?;
        let mut fresh = Run::new(make(), 0, 256);
        fresh.submit(2, turn, g, 8, CacheScope::Private);
        fresh.run();
        ensure(
            hit.outputs[&2] == fresh.outputs[&2],
            "tokens differ after a hit".into(),
        )?;
        let n = agree("hit and fresh", hit.book(), fresh.book())?;
        ensure(n >= 8, format!("only {n} prefixes compared"))?;
        Ok(format!("{n} prefixes, {cached} positions from the cache"))
    })();
    out.push(("cache_hit_and_fresh", res));

    // Greedy at every draft width.
    let res = (|| {
        let mut total = 0;
        for k in 1..=3u32 {
            let mut r = Run::new(make(), k, 256);
            r.submit(1, prompt.clone(), g, 10, keyed(1));
            r.run();
            ensure(
                r.outputs[&1] == alone.outputs[&1],
                format!("tokens differ at width {k}"),
            )?;
            total += agree(&format!("width {k}"), alone.book(), r.book())?;
        }
        Ok(format!("{total} prefixes over widths 1 to 3"))
    })();
    out.push(("draft_widths", res));
    out.push(("drafting", drafting(make, &prompt, &crowd)));
    out.push(("long_context", long_context(make)));
    out
}

/// A prompt of 1,800 positions (four of FA2's 512-key chunks at its end), prefilled in
/// chunks of 256 (whose last full chunk's attention takes two passes of the partial rows)
/// and of 100 (one pass each), then decoded: the same tokens and logits bits; and decoded
/// positions against the same positions prefilled.
pub fn long_context<D: QwenDevice>(make: &dyn Fn() -> D) -> Result<String, String> {
    let g = SamplingParams::greedy();
    let prompt = TestRng(9191).tokens(1800, VOCAB);
    let run = |chunk: u32, p: Vec<u32>, max: u32| {
        let mut r = Run::long(make(), chunk);
        r.submit(1, p, g, max, CacheScope::Private);
        r.run();
        r
    };
    let wide = run(256, prompt.clone(), 6);
    let narrow = run(100, prompt.clone(), 6);
    ensure(
        wide.outputs[&1] == narrow.outputs[&1],
        "tokens differ between prefill chunks of 256 and 100".into(),
    )?;
    let mut n = agree("chunks of 256 and 100", wide.book(), narrow.book())?;
    ensure(n >= 6, format!("only {n} prefixes compared"))?;
    let mut p = prompt;
    p.extend_from_slice(&wide.outputs[&1][..3]);
    let prefilled = run(256, p, 1);
    let m = agree("decoded and prefilled", wide.book(), prefilled.book())?;
    ensure(
        m == 1,
        format!("decoded and prefilled: {m} prefixes compared"),
    )?;
    n += m;
    Ok(format!("{n} prefixes past 1,800 positions"))
}

/// With drafting on (width 3): a greedy and a seeded sampled request each alone, beside
/// traffic arriving at varied steps and prefilled in chunks, and a second turn resumed
/// from a snapshot against the same turn computed fresh: the same tokens and logits bits.
/// A sampled row's tokens at a fixed width are a function of its prefix and seed alone,
/// since its drafts and their draws are.
fn drafting<D: QwenDevice>(
    make: &dyn Fn() -> D,
    prompt: &[u32],
    crowd: &[Vec<u32>],
) -> Result<String, String> {
    let k = 3;
    let sampled = SamplingParams::random(1.0, 777).expect("params");
    let other = TestRng(4343).tokens(14, VOCAB);
    let reqs = [
        (1u64, prompt.to_vec(), SamplingParams::greedy()),
        (2, other, sampled),
    ];
    let alone: Vec<Run<D>> = reqs
        .iter()
        .map(|(id, p, s)| {
            let mut r = Run::new(make(), k, 256);
            r.submit(*id, p.clone(), *s, 12, keyed(1));
            r.run();
            r
        })
        .collect();
    let mut total = 0;
    for chunk in [3u32, 5] {
        let mut r = Run::new(make(), k, chunk);
        for (i, c) in crowd.iter().enumerate() {
            let id = 10 + u64::try_from(i).expect("small");
            r.submit(
                id,
                c.clone(),
                super::pair::params(id),
                6 + u32::try_from(i).expect("small"),
                keyed(2),
            );
            if i == 1 {
                let (id, p, s) = &reqs[0];
                r.submit(*id, p.clone(), *s, 12, keyed(1));
            }
            if i == 3 {
                let (id, p, s) = &reqs[1];
                r.submit(*id, p.clone(), *s, 12, keyed(3));
            }
            for e in r.eng.step(r.now).expect("step") {
                r.outputs.entry(e.id).or_default().extend(&e.tokens);
            }
            r.now += 1;
        }
        r.run();
        for (a, (id, _, _)) in alone.iter().zip(&reqs) {
            ensure(
                r.outputs[id] == a.outputs[id],
                format!("request {id}: tokens differ beside traffic at chunk {chunk}"),
            )?;
            total += agree(
                &format!("request {id} at chunk {chunk}"),
                a.book(),
                r.book(),
            )?;
        }
    }
    ensure(total >= 40, format!("only {total} prefixes compared"))?;
    // A second turn of the sampled request, from its snapshot and fresh.
    let (id, p, s) = &reqs[1];
    let mut turn = p.clone();
    turn.extend_from_slice(&alone[1].outputs[id]);
    turn.extend(TestRng(78).tokens(5, VOCAB));
    let mut hit = Run::new(make(), k, 256);
    hit.submit(*id, p.clone(), *s, 12, keyed(1));
    hit.run();
    hit.submit(5, turn.clone(), *s, 10, keyed(1));
    hit.run();
    let cached = hit.cached[&5];
    ensure(cached > 0, "the second turn did not hit a snapshot".into())?;
    let mut fresh = Run::new(make(), k, 256);
    fresh.submit(5, turn, *s, 10, CacheScope::Private);
    fresh.run();
    ensure(
        hit.outputs[&5] == fresh.outputs[&5],
        "sampled tokens differ after a hit".into(),
    )?;
    let n = agree("hit and fresh", hit.book(), fresh.book())?;
    ensure(n >= 10, format!("only {n} prefixes compared after the hit"))?;
    Ok(format!(
        "{total} prefixes beside traffic, {n} after a hit of {cached} positions"
    ))
}

/// The logits of a fixed set of greedy requests (prompts of 5 to 40 tokens, 12 tokens
/// each), for comparing two profiles.
pub fn reference_book<D: QwenDevice>(make: &dyn Fn() -> D) -> Book {
    let mut rng = TestRng(9090);
    let mut r = Run::new(make(), 0, 16);
    for (i, n) in [5usize, 13, 22, 40].into_iter().enumerate() {
        let id = u64::try_from(i).expect("small");
        r.submit(
            id,
            rng.tokens(n, VOCAB),
            SamplingParams::greedy(),
            12,
            CacheScope::Private,
        );
    }
    r.run();
    r.book().clone()
}

/// How far `a`'s logits are from `b`'s where both computed a prefix: positions compared,
/// top-1 agreement over the sampleable ids, and a histogram of BF16 units in the last
/// place.
pub fn deviation(a: &Book, b: &Book) -> String {
    let sampleable = super::pair::SAMPLEABLE as usize;
    let (mut n, mut top) = (0usize, 0usize);
    let mut hist = [0usize; 6];
    let mut worst = 0f32;
    let argmax = |v: &[u32]| {
        (0..sampleable)
            .max_by(|&i, &j| {
                f32::from_bits(v[i])
                    .total_cmp(&f32::from_bits(v[j]))
                    .then(j.cmp(&i))
            })
            .unwrap_or(0)
    };
    for (k, va) in a {
        let Some(vb) = b.get(k) else {
            continue;
        };
        n += 1;
        if argmax(va) == argmax(vb) {
            top += 1;
        }
        for (x, y) in va.iter().zip(vb) {
            let hi = |w: u32| u16::try_from(w >> 16).expect("the high half");
            let d = u64::from(
                (super::cases::ord16(hi(*x)) - super::cases::ord16(hi(*y))).unsigned_abs(),
            );
            hist[usize::try_from(d.min(5)).expect("small")] += 1;
            worst = worst.max((f32::from_bits(*x) - f32::from_bits(*y)).abs());
        }
    }
    format!(
        "{n} positions, top-1 {top}/{n}, BF16 ulp 0/1/2/3/4/5+ = {hist:?}, largest |Δlogit| {worst}"
    )
}

/// A workload decoding most of its steps as uniform decode rows: a greedy request beside
/// sampled traffic of varied sizes, prefilled in chunks, at draft width `k`.
pub fn workload<D: QwenDevice>(make: &dyn Fn() -> D, k: u32) -> (Book, HashMap<u64, Vec<u32>>) {
    let mut rng = TestRng(3131);
    let mut r = Run::new(make(), k, 6);
    for (i, n) in [9usize, 4, 17, 12].into_iter().enumerate() {
        let id = u64::try_from(i).expect("small");
        let params = if i == 0 {
            SamplingParams::greedy()
        } else {
            super::pair::params(id)
        };
        r.submit(id, rng.tokens(n, VOCAB), params, 12, CacheScope::Private);
    }
    r.run();
    (r.book().clone(), r.outputs.clone())
}

/// The same workloads on every path of `paths` (eager, a rung's launches run directly, a
/// rung's graph replayed), at widths 0 and 3: the same tokens and every logit's bits.
pub fn across_paths<D: QwenDevice>(
    make: &dyn Fn() -> D,
    paths: &[(CudaGraphs, DecodePath)],
) -> Vec<(String, Result<String, String>)> {
    let mut out = Vec::new();
    for k in [0u32, 3] {
        let runs: Vec<_> = paths
            .iter()
            .map(|&(g, p)| {
                let before = super::rung_steps();
                let w = super::with_path(g, p, || workload(make, k));
                (w, super::rung_steps() - before)
            })
            .collect();
        let ((base, base_out), _) = &runs[0];
        for (&(g, p), ((book, outs), rung)) in paths.iter().zip(&runs).skip(1) {
            let res = (|| {
                ensure(
                    outs == base_out,
                    format!("tokens differ on {p:?} with graphs {}", g.as_str()),
                )?;
                let n = agree(&format!("{p:?}"), base, book)?;
                ensure(n >= 40, format!("only {n} prefixes compared"))?;
                ensure(
                    p == DecodePath::Eager || *rung > 0,
                    "no step ran a rung".into(),
                )?;
                Ok(format!("{n} prefixes, {rung} rung steps"))
            })();
            out.push((format!("width {k}: {:?} == {:?}", paths[0].1, p), res));
        }
    }
    out
}
