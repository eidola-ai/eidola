//! The scheduler: continuous batching with chunked prefill, a token budget, recompute
//! preemption, speculative decoding control flow, and request lifecycle.
//!
//! # A step
//!
//! 1. **Expiry.** If `sweep_interval_ms` has passed, expired prefix-cache entries are
//!    evicted (their blocks are zeroed by this step's maintenance).
//! 2. **Running sequences**, oldest arrival first. A decoding sequence (exactly one known
//!    token without KV) asks for `1 + k` query tokens (`k` drafts when speculative
//!    decoding is on). A prefilling sequence asks for as many of its uncomputed tokens as
//!    the remaining token budget and `max_prefill_chunk` allow. KV for the request is
//!    allocated; when the pools (after evicting unreferenced cache entries) cannot supply
//!    it, the **newest-arrived** running sequence is preempted — its blocks released (its
//!    sealed prefix stays cached) and it returns to the waiting queue — until the
//!    allocation fits or the sequence itself was the one preempted.
//! 3. **Waiting sequences**, oldest arrival first, if nothing was preempted this step:
//!    admitted while a slot, budget, and KV for their first chunk are available. Admission
//!    is strictly first-come-first-served: the first request that does not fit blocks those
//!    behind it (no starvation of long prompts).
//! 4. The batch is padded to the smallest captured [`Bucket`](crate::spec::Bucket) and
//!    executed together with all pending maintenance and table updates.
//! 5. **Commit.** Produced tokens are appended; the sequence finishes on an EOS or stop
//!    token, `max_tokens`, or the model length. KV beyond the accepted positions is rolled
//!    back, full blocks are sealed into the prefix cache, and finished sequences are
//!    released.
//!
//! Preempted sequences resume by recomputation: they are re-admitted with their prompt plus
//! the tokens generated so far, normally hitting their own sealed blocks in the prefix
//! cache. Because sampling draws are keyed by `(seed, position)` and the executor's KV is a
//! function of the token prefix alone, a preempted-and-resumed sequence produces exactly
//! the tokens an uninterrupted run would.
//!
//! The scheduler never prices anything; it reports cached prompt tokens for information
//! only.

use std::collections::{BTreeMap, HashMap};

use crate::executor::{Executor, ExecutorError, SeqEntry, StepInput};
use crate::kv::{CachePolicy, KvManager, Millis, Release};
use crate::sampling::SamplingParams;
use crate::secret::EngineSalt;
use crate::spec::{AttentionKind, Bucket, ModelSpec};

/// Caller-chosen request identifier.
pub type RequestId = u64;

/// Prefix-cache scope of a request.
#[derive(Debug)]
pub enum CacheScope {
    /// A salt derived from the client's conversation key: blocks are shared with later
    /// requests carrying the same salt, within the cache lifetime.
    Keyed(EngineSalt),
    /// No key: a fresh random salt, and everything the request sealed is purged when it
    /// finishes.
    Private,
}

/// A generation request in token space.
#[derive(Debug)]
pub struct Request {
    /// Identifier echoed in events.
    pub id: RequestId,
    /// Prompt tokens (non-empty).
    pub prompt: Vec<u32>,
    /// Sampling parameters (including the seed).
    pub sampling: SamplingParams,
    /// Maximum generated tokens (`>= 1`).
    pub max_tokens: u32,
    /// Extra token ids that end generation (in addition to the engine's EOS ids). The stop
    /// token is included in the output.
    pub stop_token_ids: Vec<u32>,
    /// Prefix-cache scope.
    pub cache: CacheScope,
}

/// Why a request finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// An EOS or stop token was produced.
    Stop,
    /// `max_tokens` or the model length was reached.
    Length,
    /// Cancelled by the caller.
    Cancelled,
}

/// Progress for one request in one step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    /// The request.
    pub id: RequestId,
    /// Tokens produced this step.
    pub tokens: Vec<u32>,
    /// Set when the request finished.
    pub finish: Option<FinishReason>,
    /// With `finish`: prompt positions served from the prefix cache at the request's first
    /// admission (usage detail for the client only; never an input to price). 0 otherwise.
    pub cached_prompt_tokens: u32,
}

/// Why a request was refused at submission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmitError {
    /// Prompt is empty.
    EmptyPrompt,
    /// Prompt plus one generated token exceeds the model length.
    TooLong,
    /// `max_tokens` is zero.
    ZeroMaxTokens,
    /// The id is already in use.
    DuplicateId,
    /// The prompt can never fit in KV memory.
    ExceedsCapacity,
    /// A prompt token is outside the sampleable vocabulary (a padded logit row, or no
    /// row at all): no tokenizer produces it.
    InvalidToken,
}

/// Why an [`Engine`] could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// The executor's [`ModelSpec`] is inconsistent.
    Spec(String),
    /// The scheduler configuration, against the executor's spec, admits a sequence
    /// the scheduler could never step (it would be planned and skipped forever).
    Scheduler(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Spec(e) => write!(f, "invalid model spec: {e}"),
            ConfigError::Scheduler(e) => write!(f, "invalid scheduler configuration: {e}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Scheduler configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Query tokens (host plus drafted) per step. Capped by the largest bucket.
    pub max_batched_tokens: u32,
    /// Sequences per step. Capped by the largest bucket and the state slots.
    pub max_seqs: u32,
    /// Largest prefill chunk for one sequence in one step.
    pub max_prefill_chunk: u32,
    /// Tokens that end every request.
    pub eos_token_ids: Vec<u32>,
    /// Use speculative decoding when the executor offers drafts.
    pub speculative: bool,
    /// Prefix-cache lifetime policy.
    pub cache: CachePolicy,
    /// Minimum interval between automatic expiry sweeps inside [`Engine::step`].
    pub sweep_interval_ms: Millis,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batched_tokens: 2048,
            max_seqs: 64,
            max_prefill_chunk: 2048,
            eos_token_ids: Vec::new(),
            speculative: true,
            cache: CachePolicy::default(),
            sweep_interval_ms: 1000,
        }
    }
}

#[derive(Debug)]
struct Sequence {
    req: Request,
    salt: EngineSalt,
    tokens: Vec<u32>,
    generated: u32,
    arrival: u64,
    cached_prompt_tokens: Option<u32>,
    preemptions: u32,
}

#[derive(Debug)]
struct Planned {
    id: RequestId,
    entry: SeqEntry,
}

/// Counters for tests and content-free metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Steps executed (including maintenance-only steps).
    pub steps: u64,
    /// Preemptions.
    pub preemptions: u64,
    /// Drafted tokens proposed.
    pub drafted: u64,
    /// Drafted tokens accepted.
    pub accepted: u64,
}

/// The host engine over one executor.
pub struct Engine<E: Executor> {
    exec: E,
    spec: ModelSpec,
    config: SchedulerConfig,
    kv: KvManager,
    seqs: HashMap<RequestId, Sequence>,
    waiting: BTreeMap<u64, RequestId>,
    running: BTreeMap<u64, RequestId>,
    next_arrival: u64,
    last_sweep: Option<Millis>,
    stats: Stats,
    largest: Bucket,
}

impl<E: Executor> std::fmt::Debug for Engine<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("waiting", &self.waiting.len())
            .field("running", &self.running.len())
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl<E: Executor> Engine<E> {
    /// An engine over `exec`.
    ///
    /// Refuses an inconsistent spec, and any scheduler configuration under which a
    /// sequence could never be stepped: every step must have room for at least one
    /// sequence, a prefill chunk of at least one token, and one decode row at its full
    /// `1 + k` cost (`k` drafts when speculation is on). Speculation is never silently
    /// turned off to make a configuration fit.
    pub fn new(exec: E, config: SchedulerConfig) -> Result<Self, ConfigError> {
        let spec = exec.spec().clone();
        spec.validate().map_err(ConfigError::Spec)?;
        let largest = *spec.buckets.last().expect("validated: at least one bucket");
        let kv = KvManager::new(&spec, config.cache);
        let engine = Self {
            exec,
            spec,
            config,
            kv,
            seqs: HashMap::new(),
            waiting: BTreeMap::new(),
            running: BTreeMap::new(),
            next_arrival: 0,
            last_sweep: None,
            stats: Stats::default(),
            largest,
        };
        engine.check_progress()?;
        Ok(engine)
    }

    /// The scheduler half of [`Engine::new`]'s validation.
    fn check_progress(&self) -> Result<(), ConfigError> {
        let err = |m: String| Err(ConfigError::Scheduler(m));
        if self.seat_budget() == 0 {
            return err(format!(
                "no sequence fits a step (max_seqs {}, largest bucket {} sequences)",
                self.config.max_seqs, self.largest.max_seqs
            ));
        }
        if self.config.max_prefill_chunk == 0 {
            return err("max_prefill_chunk must be non-zero".into());
        }
        let decode_row = 1 + self.drafts();
        if self.token_budget() < decode_row {
            return err(format!(
                "a decode row costs {decode_row} query tokens (1 + {} drafts) but a step \
                 holds {} (max_batched_tokens {}, largest bucket {} tokens)",
                self.drafts(),
                self.token_budget(),
                self.config.max_batched_tokens,
                self.largest.max_tokens
            ));
        }
        Ok(())
    }

    /// The executor (tests inspect the mock through this).
    pub fn executor(&self) -> &E {
        &self.exec
    }

    /// The KV manager (tests check invariants through this).
    pub fn kv(&self) -> &KvManager {
        &self.kv
    }

    /// Counters.
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Requests not yet finished.
    pub fn unfinished(&self) -> usize {
        self.seqs.len()
    }

    /// Cached prompt tokens at a request's first admission (informational; never priced).
    pub fn cached_prompt_tokens(&self, id: RequestId) -> Option<u32> {
        self.seqs.get(&id).and_then(|s| s.cached_prompt_tokens)
    }

    fn token_budget(&self) -> u32 {
        self.config.max_batched_tokens.min(self.largest.max_tokens)
    }

    fn seat_budget(&self) -> u32 {
        self.config
            .max_seqs
            .min(self.largest.max_seqs)
            .min(self.spec.num_state_slots)
    }

    fn drafts(&self) -> u32 {
        if self.config.speculative {
            self.spec.max_draft_tokens
        } else {
            0
        }
    }

    /// Queues a request.
    pub fn submit(&mut self, req: Request) -> Result<(), SubmitError> {
        if req.prompt.is_empty() {
            return Err(SubmitError::EmptyPrompt);
        }
        if req.max_tokens == 0 {
            return Err(SubmitError::ZeroMaxTokens);
        }
        if req.prompt.len() as u32 >= self.spec.max_model_len {
            return Err(SubmitError::TooLong);
        }
        if self.seqs.contains_key(&req.id) {
            return Err(SubmitError::DuplicateId);
        }
        if req
            .prompt
            .iter()
            .any(|&t| t >= self.spec.sampleable_vocab_size)
        {
            return Err(SubmitError::InvalidToken);
        }
        // The prompt (plus the first generated token) must fit in every full-attention pool
        // on its own; sliding-window pools are bounded by the window instead.
        let need = self.spec.blocks_for(req.prompt.len() as u32 + 1);
        if self
            .spec
            .kv_groups
            .iter()
            .any(|g| g.attention == AttentionKind::Full && g.num_blocks - 1 < need)
        {
            return Err(SubmitError::ExceedsCapacity);
        }
        let salt = match &req.cache {
            CacheScope::Keyed(s) => s.clone(),
            CacheScope::Private => EngineSalt::fresh(),
        };
        let arrival = self.next_arrival;
        self.next_arrival += 1;
        let id = req.id;
        let tokens = req.prompt.clone();
        self.seqs.insert(
            id,
            Sequence {
                req,
                salt,
                tokens,
                generated: 0,
                arrival,
                cached_prompt_tokens: None,
                preemptions: 0,
            },
        );
        self.waiting.insert(arrival, id);
        Ok(())
    }

    /// Cancels a request. Returns its final event if it was unfinished.
    pub fn cancel(&mut self, id: RequestId, now: Millis) -> Option<Event> {
        let seq = self.seqs.remove(&id)?;
        if self.running.remove(&seq.arrival).is_some() {
            self.kv
                .release(id, Release::Finish, seq.req.prompt.len() as u32, now);
        } else {
            self.waiting.remove(&seq.arrival);
            self.kv.forget(id);
        }
        Some(Event {
            id,
            tokens: Vec::new(),
            finish: Some(FinishReason::Cancelled),
            cached_prompt_tokens: seq.cached_prompt_tokens.unwrap_or(0),
        })
    }

    /// Evicts expired prefix-cache entries now and, if that (or anything else) left
    /// maintenance pending, runs a maintenance-only step so freed memory is zeroed without
    /// waiting for traffic.
    pub fn sweep(&mut self, now: Millis) -> Result<(), ExecutorError> {
        self.kv.sweep(now);
        self.last_sweep = Some(now);
        if self.kv.has_pending_maintenance() {
            let (maintenance, table_updates) = self.kv.take_pending();
            let step = StepInput {
                bucket: self.spec.buckets[0],
                maintenance,
                table_updates,
                seqs: Vec::new(),
                token_ids: Vec::new(),
                positions: Vec::new(),
                return_logits: false,
            };
            self.exec.execute(&step)?;
            self.stats.steps += 1;
        }
        Ok(())
    }

    fn preempt(&mut self, id: RequestId, now: Millis) {
        let seq = self.seqs.get_mut(&id).expect("running sequence");
        self.running.remove(&seq.arrival);
        self.waiting.insert(seq.arrival, id);
        seq.preemptions += 1;
        self.stats.preemptions += 1;
        let prompt_len = seq.req.prompt.len() as u32;
        self.kv.release(id, Release::Preempt, prompt_len, now);
    }

    /// Plans and executes one step. Returns the events it produced (empty when idle).
    pub fn step(&mut self, now: Millis) -> Result<Vec<Event>, ExecutorError> {
        if self
            .last_sweep
            .is_none_or(|t| now.saturating_sub(t) >= self.config.sweep_interval_ms)
        {
            self.kv.sweep(now);
            self.last_sweep = Some(now);
        }

        let mut budget = self.token_budget();
        let mut seats = self.seat_budget();
        let mut planned: Vec<Planned> = Vec::new();
        let mut events: Vec<Event> = Vec::new();
        let mut preempted_any = false;
        let k = self.drafts();

        // Running sequences, oldest first.
        let order: Vec<(u64, RequestId)> = self.running.iter().map(|(a, i)| (*a, *i)).collect();
        for (arrival, id) in order {
            if !self.running.contains_key(&arrival) {
                continue; // preempted earlier in this loop
            }
            if seats == 0 || budget == 0 {
                break;
            }
            let Some((entry, cost)) = self.plan_row(id, budget, k) else {
                continue;
            };
            let upto = entry.context_len + entry.num_tokens + entry.num_drafts;
            let mut scheduled = true;
            while !self.kv.allocate(id, upto) {
                if self.running.len() == 1 {
                    // Alone, with every unreferenced cache entry already evicted: this
                    // sequence can never grow further.
                    events.push(self.finish_now(id, FinishReason::Length, now));
                    scheduled = false;
                    break;
                }
                let (&victim_arrival, &victim) =
                    self.running.iter().next_back().expect("self is running");
                self.preempt(victim, now);
                preempted_any = true;
                if victim_arrival == arrival {
                    scheduled = false;
                    break;
                }
            }
            if scheduled {
                budget -= cost;
                seats -= 1;
                planned.push(Planned { id, entry });
            }
        }

        // Waiting sequences, strictly FCFS.
        if !preempted_any {
            while seats > 0 && budget > 0 && self.kv.has_free_slot() {
                let Some((&arrival, &id)) = self.waiting.iter().next() else {
                    break;
                };
                let seq = &self.seqs[&id];
                let Some(cached) = self.kv.admit(
                    id,
                    seq.salt.clone(),
                    matches!(seq.req.cache, CacheScope::Keyed(_)),
                    &seq.tokens,
                    seq.req.prompt.len() as u32,
                    now,
                ) else {
                    break;
                };
                let Some((entry, cost)) = self.plan_row(id, budget, k) else {
                    self.kv.release(
                        id,
                        Release::Preempt,
                        self.seqs[&id].req.prompt.len() as u32,
                        now,
                    );
                    break;
                };
                let upto = entry.context_len + entry.num_tokens + entry.num_drafts;
                if !self.kv.allocate(id, upto) {
                    self.kv.release(
                        id,
                        Release::Preempt,
                        self.seqs[&id].req.prompt.len() as u32,
                        now,
                    );
                    if self.running.is_empty() && planned.is_empty() {
                        // Nothing else holds memory: it can never be admitted.
                        self.waiting.remove(&arrival);
                        let seq = self.seqs.remove(&id).expect("waiting sequence");
                        self.kv.forget(id);
                        events.push(Event {
                            id,
                            tokens: Vec::new(),
                            finish: Some(FinishReason::Length),
                            cached_prompt_tokens: seq.cached_prompt_tokens.unwrap_or(0),
                        });
                    }
                    break;
                }
                let seq = self.seqs.get_mut(&id).expect("waiting sequence");
                if seq.cached_prompt_tokens.is_none() {
                    seq.cached_prompt_tokens = Some(cached);
                }
                self.waiting.remove(&arrival);
                self.running.insert(arrival, id);
                budget -= cost;
                seats -= 1;
                planned.push(Planned { id, entry });
            }
        }

        if planned.is_empty() && !self.kv.has_pending_maintenance() {
            return Ok(events);
        }
        events.extend(self.execute_and_commit(planned, now)?);
        Ok(events)
    }

    fn finish_now(&mut self, id: RequestId, reason: FinishReason, now: Millis) -> Event {
        let seq = self.seqs.remove(&id).expect("running sequence");
        self.running.remove(&seq.arrival);
        self.kv
            .release(id, Release::Finish, seq.req.prompt.len() as u32, now);
        Event {
            id,
            tokens: Vec::new(),
            finish: Some(reason),
            cached_prompt_tokens: seq.cached_prompt_tokens.unwrap_or(0),
        }
    }

    /// Plans the row for a running (or just admitted) sequence within `budget` query tokens.
    fn plan_row(&self, id: RequestId, budget: u32, k: u32) -> Option<(SeqEntry, u32)> {
        let seq = &self.seqs[&id];
        let computed = self.kv.computed(id);
        let len = seq.tokens.len() as u32;
        let pending = len - computed;
        debug_assert!(pending >= 1);
        let (num_tokens, num_drafts) = if pending == 1 {
            // Never draft past the model length: the last written position is
            // `computed + drafts` and the output may grow by `drafts + 1`.
            let room = self.spec.max_model_len.saturating_sub(len);
            let d = k.min(room.saturating_sub(1));
            (1, d)
        } else {
            (pending.min(self.config.max_prefill_chunk), 0)
        };
        let cost = num_tokens + num_drafts;
        if cost > budget {
            if pending == 1 {
                return None;
            }
            let n = budget;
            return Some((self.entry(id, computed, n, 0, n == pending), n));
        }
        let sample = computed + num_tokens == len;
        Some((
            self.entry(id, computed, num_tokens, num_drafts, sample),
            cost,
        ))
    }

    fn entry(&self, id: RequestId, computed: u32, n: u32, drafts: u32, sample: bool) -> SeqEntry {
        SeqEntry {
            slot: self.kv.slot(id),
            token_start: 0,
            num_tokens: n,
            context_len: computed,
            num_drafts: if sample { drafts } else { 0 },
            sample,
            sampling: self.seqs[&id].req.sampling,
        }
    }

    fn execute_and_commit(
        &mut self,
        mut planned: Vec<Planned>,
        now: Millis,
    ) -> Result<Vec<Event>, ExecutorError> {
        let mut token_ids = Vec::new();
        let mut positions = Vec::new();
        for p in &mut planned {
            let seq = &self.seqs[&p.id];
            p.entry.token_start = token_ids.len() as u32;
            let start = p.entry.context_len as usize;
            let end = start + p.entry.num_tokens as usize;
            token_ids.extend_from_slice(&seq.tokens[start..end]);
            positions.extend(p.entry.context_len..p.entry.context_len + p.entry.num_tokens);
        }
        let rows = planned.len() as u32;
        let query: u32 = planned
            .iter()
            .map(|p| p.entry.num_tokens + p.entry.num_drafts)
            .sum();
        let bucket = self
            .spec
            .bucket_for(rows, query)
            .expect("budgets keep every batch within the largest bucket");
        let (maintenance, table_updates) = self.kv.take_pending();
        let step = StepInput {
            bucket,
            maintenance,
            table_updates,
            seqs: planned.iter().map(|p| p.entry).collect(),
            token_ids,
            positions,
            return_logits: false,
        };
        let out = self.exec.execute(&step)?;
        self.stats.steps += 1;
        // A padded logit row has no token: an executor that returns one has broken the
        // seam contract, and nothing it produced in this step is committed.
        let limit = self.spec.sampleable_vocab_size;
        for row in 0..planned.len() {
            if let Some(&t) = out.row(row).iter().find(|&&t| t >= limit) {
                return Err(ExecutorError(format!(
                    "returned token {t} outside the sampleable vocabulary ({limit})"
                )));
            }
        }

        let mut events = Vec::new();
        for (row, p) in planned.iter().enumerate() {
            let produced = out.row(row).to_vec();
            let e = p.entry;
            if e.sample && e.num_drafts > 0 {
                self.stats.drafted += e.num_drafts as u64;
                self.stats.accepted += produced.len() as u64 - 1;
            }
            let id = p.id;
            let seq = self.seqs.get_mut(&id).expect("planned sequence");
            let mut emitted = Vec::new();
            let mut finish = None;
            for &t in &produced {
                seq.tokens.push(t);
                seq.generated += 1;
                emitted.push(t);
                if self.config.eos_token_ids.contains(&t) || seq.req.stop_token_ids.contains(&t) {
                    finish = Some(FinishReason::Stop);
                } else if seq.generated >= seq.req.max_tokens
                    || seq.tokens.len() as u32 >= self.spec.max_model_len
                {
                    finish = Some(FinishReason::Length);
                }
                if finish.is_some() {
                    break;
                }
            }
            // KV is valid for the host tokens plus the accepted drafts.
            let accepted_drafts = produced.len().saturating_sub(1) as u32;
            let mut computed = e.context_len + e.num_tokens;
            if e.sample {
                computed += accepted_drafts;
            }
            let computed = computed.min(seq.tokens.len() as u32 - 1).max(e.context_len);
            let prompt_len = seq.req.prompt.len() as u32;
            self.kv.commit(id, computed, &seq.tokens, now);
            if let Some(reason) = finish {
                self.kv.release(id, Release::Finish, prompt_len, now);
                let seq = self.seqs.remove(&id).expect("sequence");
                self.running.remove(&seq.arrival);
                events.push(Event {
                    id,
                    tokens: emitted,
                    finish: Some(reason),
                    cached_prompt_tokens: seq.cached_prompt_tokens.unwrap_or(0),
                });
            } else if !emitted.is_empty() {
                events.push(Event {
                    id,
                    tokens: emitted,
                    finish: None,
                    cached_prompt_tokens: 0,
                });
            }
        }
        Ok(events)
    }
}
