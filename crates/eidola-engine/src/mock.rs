//! A deterministic mock executor: a hash-based fake language model that really stores and
//! reads paged KV through the block tables it is given, plus a fake drafter whose agreement
//! with the target is controllable.
//!
//! The "KV" of position `p` holding token `t` in group `g` is `mix(t, p, g) | 1`, stored in
//! the physical slot the tables select, together with `p`. A target query at `q` folds the
//! KV values its groups' attention can see (all of `0..=q` for full attention, the window
//! for sliding groups) into a state that seeds the logits. Every read asserts that the slot
//! is mapped, non-zero, and holds position `q`'s data — so a wrong block table, a missing
//! sliding window, a premature free, an unexecuted zero, or a stale prefix hit panics or
//! changes the output, while correct paging reproduces [`reference_logits`] exactly.
//!
//! Writes assert that the target block is mapped by exactly one (slot, index) pair: blocks
//! shared through the prefix cache are immutable.

use std::collections::HashMap;

use crate::executor::{Executor, ExecutorError, Maintenance, StepInput, StepOutput};
use crate::sampling::{self, SamplingParams, Stream, mix64};
use crate::spec::{AttentionKind, Bucket, KvGroupSpec, KvRole, ModelSpec, NULL_BLOCK};

/// Mock model parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MockConfig {
    /// Model identity: different seeds are different "models".
    pub model_seed: u64,
    /// Logits are uniform in `[-spread / 2, spread / 2]`.
    pub logit_spread: f32,
    /// Drafter mixture weight on the target distribution, in `[0, 1]`: 1 drafts exactly
    /// from the target (everything accepted), 0 drafts from unrelated noise.
    pub draft_agreement: f64,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            model_seed: 0x5eed,
            logit_spread: 4.0,
            draft_agreement: 0.7,
        }
    }
}

/// A small MiMo-shaped spec: one full-attention group, one sliding-window target group,
/// and one sliding-window drafter-context group.
pub fn mimo_like_spec(
    vocab: u32,
    block_size: u32,
    window: u32,
    drafter_window: u32,
    num_blocks: u32,
    slots: u32,
    k: u32,
) -> ModelSpec {
    let group = |name: &str, role, attention, heads| KvGroupSpec {
        name: name.into(),
        role,
        attention,
        num_layers: 1,
        num_kv_heads: heads,
        head_dim_qk: 192,
        head_dim_v: 128,
        num_blocks,
    };
    ModelSpec {
        vocab_size: vocab,
        sampleable_vocab_size: vocab,
        block_size,
        max_model_len: 4096,
        kv_groups: vec![
            group("global", KvRole::Target, AttentionKind::Full, 4),
            group(
                "sliding",
                KvRole::Target,
                AttentionKind::Sliding { window },
                8,
            ),
            group(
                "drafter",
                KvRole::Drafter,
                AttentionKind::Sliding {
                    window: drafter_window,
                },
                8,
            ),
        ],
        max_draft_tokens: k,
        num_state_slots: slots,
        buckets: vec![
            Bucket {
                max_seqs: 1,
                max_tokens: 16,
            },
            Bucket {
                max_seqs: 8,
                max_tokens: 64,
            },
            Bucket {
                max_seqs: 32,
                max_tokens: 256,
            },
        ],
    }
}

/// KV value for `token` at `pos` in `group` (never zero).
pub fn kv_value(cfg: &MockConfig, token: u32, pos: u32, group: u32) -> u64 {
    mix64(cfg.model_seed ^ mix64(((token as u64) << 32) ^ ((pos as u64) << 4) ^ group as u64)) | 1
}

fn fold_state(
    spec: &ModelSpec,
    role: KvRole,
    q: u32,
    mut read: impl FnMut(u32, u32) -> u64,
) -> u64 {
    let mut acc = 0x9e37_79b9_7f4a_7c15u64;
    for (g, gs) in spec.kv_groups.iter().enumerate() {
        if gs.role != role {
            continue;
        }
        for pos in gs.attention.first_visible(q)..=q {
            acc = mix64(acc ^ read(g as u32, pos) ^ ((g as u64) << 56));
        }
    }
    acc
}

fn logits_from_state(cfg: &MockConfig, vocab: u32, state: u64) -> Vec<f32> {
    (0..vocab as u64)
        .map(|i| {
            let r = (mix64(state ^ (i + 1).wrapping_mul(0xd6e8_feb8_6659_fd93)) >> 40) as f32
                / (1u32 << 24) as f32;
            (r - 0.5) * cfg.logit_spread
        })
        .collect()
}

/// The target logits for the token after position `q` given the full token history — the
/// dense oracle the paged mock must reproduce.
pub fn reference_logits(spec: &ModelSpec, cfg: &MockConfig, tokens: &[u32], q: u32) -> Vec<f32> {
    let state = fold_state(spec, KvRole::Target, q, |g, p| {
        kv_value(cfg, tokens[p as usize], p, g)
    });
    logits_from_state(cfg, spec.vocab_size, state)
}

/// Non-speculative generation straight from [`reference_logits`]: what every engine run must
/// reproduce exactly (and what a speculative run must match in distribution).
pub fn reference_generate(
    spec: &ModelSpec,
    cfg: &MockConfig,
    prompt: &[u32],
    params: &SamplingParams,
    max_tokens: u32,
    stop: &[u32],
) -> Vec<u32> {
    let mut tokens = prompt.to_vec();
    let mut out = Vec::new();
    while (out.len() as u32) < max_tokens && (tokens.len() as u32) < spec.max_model_len {
        let q = tokens.len() as u32 - 1;
        let logits = reference_logits(spec, cfg, &tokens, q);
        let t = sampling::sample(spec.logits(&logits), params, q as u64 + 1);
        tokens.push(t);
        out.push(t);
        if stop.contains(&t) {
            break;
        }
    }
    out
}

/// The mock executor.
#[derive(Debug)]
pub struct MockExecutor {
    spec: ModelSpec,
    cfg: MockConfig,
    /// `[group][block * block_size + offset] = (value, position)`; value 0 means zeroed.
    memory: Vec<Vec<(u64, u32)>>,
    /// `[slot][group][index]`.
    tables: Vec<Vec<Vec<u32>>>,
    /// `[group]`: block -> number of (slot, index) mappings.
    mapped: Vec<HashMap<u32, u32>>,
    zero_log: Vec<(u64, u32, u32)>,
    steps: u64,
    buckets_used: Vec<Bucket>,
}

impl MockExecutor {
    /// A mock over `spec`.
    pub fn new(spec: ModelSpec, cfg: MockConfig) -> Self {
        let bs = spec.block_size as usize;
        let width = spec.max_blocks_per_seq() as usize;
        let groups = spec.kv_groups.len();
        Self {
            memory: spec
                .kv_groups
                .iter()
                .map(|g| vec![(0, 0); g.num_blocks as usize * bs])
                .collect(),
            tables: vec![vec![vec![NULL_BLOCK; width]; groups]; spec.num_state_slots as usize],
            mapped: vec![HashMap::new(); groups],
            zero_log: Vec::new(),
            steps: 0,
            buckets_used: Vec::new(),
            spec,
            cfg,
        }
    }

    /// Every executed zero as `(step, group, block)`.
    pub fn zero_log(&self) -> &[(u64, u32, u32)] {
        &self.zero_log
    }

    /// Steps executed.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Buckets of the executed steps, in order.
    pub fn buckets_used(&self) -> &[Bucket] {
        &self.buckets_used
    }

    /// Whether every byte of a block is zero.
    pub fn block_is_zero(&self, group: u32, block: u32) -> bool {
        let bs = self.spec.block_size as usize;
        let start = block as usize * bs;
        self.memory[group as usize][start..start + bs]
            .iter()
            .all(|&(v, p)| v == 0 && p == 0)
    }

    fn set_table(&mut self, slot: u32, group: u32, index: u32, block: u32) {
        let cur = &mut self.tables[slot as usize][group as usize][index as usize];
        let old = *cur;
        *cur = block;
        let mapped = &mut self.mapped[group as usize];
        if old != NULL_BLOCK {
            let c = mapped.get_mut(&old).expect("mapped block counted");
            *c -= 1;
            if *c == 0 {
                mapped.remove(&old);
            }
        }
        if block != NULL_BLOCK {
            *mapped.entry(block).or_default() += 1;
        }
    }

    fn locate(&self, slot: u32, group: u32, pos: u32) -> Result<usize, ExecutorError> {
        let bs = self.spec.block_size;
        let block = self.tables[slot as usize][group as usize][(pos / bs) as usize];
        if block == NULL_BLOCK {
            return Err(ExecutorError(format!(
                "slot {slot} group {group} position {pos}: unmapped block"
            )));
        }
        Ok((block * bs + pos % bs) as usize)
    }

    fn write(&mut self, slot: u32, pos: u32, token: u32) -> Result<(), ExecutorError> {
        let bs = self.spec.block_size;
        for g in 0..self.spec.kv_groups.len() as u32 {
            let block = self.tables[slot as usize][g as usize][(pos / bs) as usize];
            if self.mapped[g as usize].get(&block).copied().unwrap_or(0) != 1 {
                return Err(ExecutorError(format!(
                    "slot {slot} group {g} position {pos}: write to an unmapped or shared block"
                )));
            }
            let at = self.locate(slot, g, pos)?;
            self.memory[g as usize][at] = (kv_value(&self.cfg, token, pos, g), pos);
        }
        Ok(())
    }

    fn read(&self, slot: u32, group: u32, pos: u32) -> Result<u64, ExecutorError> {
        let at = self.locate(slot, group, pos)?;
        let (v, p) = self.memory[group as usize][at];
        if v == 0 || p != pos {
            return Err(ExecutorError(format!(
                "slot {slot} group {group} position {pos}: read of zeroed or foreign KV"
            )));
        }
        Ok(v)
    }

    fn state(&self, slot: u32, role: KvRole, q: u32) -> Result<u64, ExecutorError> {
        let mut err = None;
        let s = fold_state(&self.spec, role, q, |g, p| match self.read(slot, g, p) {
            Ok(v) => v,
            Err(e) => {
                err.get_or_insert(e);
                0
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(s),
        }
    }

    fn target_probs(
        &self,
        slot: u32,
        q: u32,
        params: &SamplingParams,
    ) -> Result<(Vec<f32>, Vec<f64>), ExecutorError> {
        let logits = logits_from_state(
            &self.cfg,
            self.spec.vocab_size,
            self.state(slot, KvRole::Target, q)?,
        );
        let probs = sampling::processed_probs(self.spec.logits(&logits), params);
        Ok((logits, probs))
    }

    fn draft_probs(
        &self,
        slot: u32,
        q: u32,
        target: &[f64],
        params: &SamplingParams,
    ) -> Result<Vec<f64>, ExecutorError> {
        let has_drafter = self
            .spec
            .kv_groups
            .iter()
            .any(|g| g.role == KvRole::Drafter);
        let noise_state = if has_drafter {
            self.state(slot, KvRole::Drafter, q)?
        } else {
            mix64(q as u64 ^ 0xabcdef)
        };
        let noise_logits = logits_from_state(&self.cfg, self.spec.vocab_size, noise_state ^ 0x77);
        let noise = sampling::processed_probs(
            self.spec.logits(&noise_logits),
            &SamplingParams::random(1.0, 0),
        );
        let a = self.cfg.draft_agreement;
        let mix: Vec<f64> = target
            .iter()
            .zip(&noise)
            .map(|(p, r)| a * p + (1.0 - a) * r)
            .collect();
        if params.is_greedy() {
            let mut best = 0;
            for (i, &v) in mix.iter().enumerate() {
                if v > mix[best] {
                    best = i;
                }
            }
            let mut one_hot = vec![0.0; mix.len()];
            one_hot[best] = 1.0;
            Ok(one_hot)
        } else {
            Ok(mix)
        }
    }
}

impl Executor for MockExecutor {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
        self.steps += 1;
        if step.seqs.len() as u32 > step.bucket.max_seqs
            || step.query_tokens() > step.bucket.max_tokens
        {
            return Err(ExecutorError("batch exceeds its bucket".into()));
        }
        self.buckets_used.push(step.bucket);
        let bs = self.spec.block_size as usize;
        for m in &step.maintenance {
            match *m {
                Maintenance::Zero { group, block } => {
                    let start = block as usize * bs;
                    self.memory[group as usize][start..start + bs].fill((0, 0));
                    self.zero_log.push((self.steps, group, block));
                }
                Maintenance::Copy { group, src, dst } => {
                    let g = &mut self.memory[group as usize];
                    g.copy_within(
                        src as usize * bs..(src as usize + 1) * bs,
                        dst as usize * bs,
                    );
                }
                Maintenance::ResetSlot { slot } => {
                    for g in 0..self.spec.kv_groups.len() {
                        for idx in 0..self.tables[slot as usize][g].len() {
                            if self.tables[slot as usize][g][idx] != NULL_BLOCK {
                                self.set_table(slot, g as u32, idx as u32, NULL_BLOCK);
                            }
                        }
                    }
                }
            }
        }
        for u in &step.table_updates {
            self.set_table(u.slot, u.group, u.index, u.block);
        }

        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; step.seqs.len() * stride as usize],
            stride,
            num_tokens: vec![0; step.seqs.len()],
            logits: step.return_logits.then(Vec::new),
        };
        for (row, e) in step.seqs.iter().enumerate() {
            let toks =
                &step.token_ids[e.token_start as usize..(e.token_start + e.num_tokens) as usize];
            for (i, &t) in toks.iter().enumerate() {
                let pos = e.context_len + i as u32;
                if step.positions[e.token_start as usize + i] != pos {
                    return Err(ExecutorError("positions disagree with context_len".into()));
                }
                self.write(e.slot, pos, t)?;
            }
            // Every query must see a complete window (validates the tables).
            for i in 0..e.num_tokens {
                self.state(e.slot, KvRole::Target, e.context_len + i)?;
            }
            let mut row_logits = Vec::new();
            if !e.sample {
                if let Some(l) = out.logits.as_mut() {
                    l.push(row_logits);
                }
                continue;
            }
            let last = e.context_len + e.num_tokens - 1;
            if last == 0 && e.num_drafts > 0 {
                return Err(ExecutorError(
                    "drafts requested at position 0, where none can exist".into(),
                ));
            }
            let params = e.sampling;
            let produced = if e.num_drafts == 0 {
                let (logits, _) = self.target_probs(e.slot, last, &params)?;
                let t = sampling::sample(self.spec.logits(&logits), &params, last as u64 + 1);
                row_logits.push(logits);
                vec![t]
            } else {
                let k = e.num_drafts as usize;
                let mut drafts = Vec::with_capacity(k);
                let mut q_rows = Vec::with_capacity(k);
                for j in 0..k {
                    let q = last + j as u32;
                    let (_, p) = self.target_probs(e.slot, q, &params)?;
                    let qd = self.draft_probs(e.slot, q, &p, &params)?;
                    let d = if params.is_greedy() {
                        qd.iter().position(|&v| v == 1.0).expect("one-hot") as u32
                    } else {
                        sampling::sample_from(
                            &qd,
                            sampling::uniform(params.seed, q as u64 + 1, Stream::Draft),
                        )
                    };
                    self.write(e.slot, q + 1, d)?;
                    drafts.push(d);
                    q_rows.push(qd);
                }
                let mut p_rows = Vec::with_capacity(k + 1);
                for j in 0..=k {
                    let (logits, p) = self.target_probs(e.slot, last + j as u32, &params)?;
                    row_logits.push(logits);
                    p_rows.push(p);
                }
                let accepted =
                    sampling::chain_accept(&p_rows, &q_rows, &drafts, &params, last as u64 + 1);
                row_logits.truncate(accepted.len());
                accepted
            };
            let base = row * stride as usize;
            out.tokens[base..base + produced.len()].copy_from_slice(&produced);
            out.num_tokens[row] = produced.len() as u32;
            if let Some(l) = out.logits.as_mut() {
                l.push(row_logits);
            }
        }
        Ok(out)
    }
}
