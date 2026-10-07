//! [`CpuExecutor`]: the serving core's [`Executor`] over the reference numerics.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use eidola_engine::executor::{
    Executor, ExecutorError, Maintenance, SeqEntry, Slot, StepInput, StepOutput,
};
use eidola_engine::sampling::{self, Logits, SamplingParams, Stream};
use eidola_engine::spec::{AttentionKind, Bucket, KvGroupSpec, KvRole, ModelSpec, NULL_BLOCK};
use eidola_engine_model::ReferenceModel;
use eidola_engine_model::attention::{apply_rope, attend, rope_cos_sin};
use eidola_engine_model::config::{AttentionKind as ModelAttention, AttentionSpec};
use eidola_engine_model::tensor::{Matrix, add_assign, linear, rms_norm_rows};
use eidola_engine_model::weights::{AttentionWeights, FfnWeights};
use rayon::prelude::*;

use crate::pool::{GroupLayout, Pool, Tag};

/// Which hidden state feeds an MTP layer: the main model's into depth 0, and each depth's
/// own output into the next.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MtpHidden {
    /// The state after the final norm (the main model's `norm`, each MTP layer's
    /// `final_layernorm`): what vLLM and SGLang feed.
    #[default]
    Normed,
    /// The state before it: what llama.cpp feeds.
    PreNorm,
}

/// Executor configuration: memory geometry and drafter choices. The model fixes the rest.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuExecutorConfig {
    /// Positions per KV block (every group).
    pub block_size: u32,
    /// Physical blocks per group, including the null block 0.
    pub num_blocks: u32,
    /// Per-sequence state rows.
    pub num_state_slots: u32,
    /// Longest sequence.
    pub max_model_len: u32,
    /// Captured step shapes, ascending.
    pub buckets: Vec<Bucket>,
    /// The loaded MTP layer serving each draft depth; its length is the draft width `k`.
    /// Empty disables drafting (and the drafter KV group).
    pub mtp_depths: Vec<usize>,
    /// Hidden-state chaining between the main model and the MTP depths.
    pub mtp_hidden: MtpHidden,
    /// Token ids the model may emit (the tokenizer's vocabulary, added tokens included);
    /// the head's rows past it are padding and take no part in sampling, drafting or
    /// acceptance. Reported as [`ModelSpec::sampleable_vocab_size`]. It comes from the
    /// tokenizer, never from the weights.
    pub sampleable_vocab_size: u32,
    /// Run the target forward over the batch padded to the bucket's token count. Padding
    /// rows go through every row-wise kernel and are discarded; with row-independent
    /// kernels they cannot change any real row.
    pub pad_batches: bool,
    /// Testing aid: compute logits at every computed position and keep a [`RowRecord`] of
    /// every row (see [`CpuExecutor::take_records`]).
    pub record: bool,
}

impl CpuExecutorConfig {
    /// A configuration drafting with every loaded MTP layer, at most three, emitting ids
    /// below `sampleable_vocab_size` (the tokenizer's vocabulary size).
    pub fn for_model(model: &ReferenceModel, sampleable_vocab_size: u32) -> Self {
        Self {
            block_size: 16,
            num_blocks: 256,
            num_state_slots: 16,
            max_model_len: model.weights.config.max_position_embeddings as u32,
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
            mtp_depths: (0..model.weights.mtp.len().min(3)).collect(),
            mtp_hidden: MtpHidden::Normed,
            sampleable_vocab_size,
            pad_batches: false,
            record: false,
        }
    }
}

/// Everything one row of one step computed, for checking against the dense oracle.
#[derive(Clone, Debug, PartialEq)]
pub struct RowRecord {
    /// The row's state slot.
    pub slot: Slot,
    /// Positions already in KV before the step.
    pub context_len: u32,
    /// Host tokens in the row.
    pub num_tokens: u32,
    /// Tokens at positions `0 ..= p + k`: the prefix read back through the block tables,
    /// this step's host tokens, then this step's drafts.
    pub tokens: Vec<u32>,
    /// Target logits by position: every host position, and every drafted position.
    pub target_logits: Vec<(u32, Vec<f32>)>,
    /// Drafter logits by `(depth, slot)` for every drafter row computed in the step.
    pub drafter_logits: Vec<(usize, u32, Vec<f32>)>,
    /// Drafted tokens.
    pub drafts: Vec<u32>,
    /// Tokens the row produced.
    pub produced: Vec<u32>,
    /// The row's sampling parameters.
    pub sampling: SamplingParams,
}

/// Per-slot model state: the hidden states the drafter continues from, one per chain level
/// (level 0 is the main model's, level `d` is MTP depth `d - 1`'s), all at position `at`.
///
/// The buffers are allocated once, at full size, and only ever overwritten in place, like
/// device-resident slot state: storing a new state and `ResetSlot` both write into the
/// same memory, so no request-derived activations are left behind in a dropped buffer.
#[derive(Clone, Debug)]
struct SlotState {
    at: Option<u32>,
    levels: Vec<Vec<f32>>,
}

impl SlotState {
    fn zeroed(levels: usize, hidden: usize) -> Self {
        Self {
            at: None,
            levels: vec![vec![0.0; hidden]; levels],
        }
    }

    /// Overwrites every level with zeros and forgets the position.
    fn scrub(&mut self) {
        self.at = None;
        for l in &mut self.levels {
            l.fill(0.0);
        }
    }
}

/// One query row of a paged attention call.
#[derive(Clone, Copy, Debug)]
struct AttnRow {
    slot: Slot,
    /// KV position written and attended from.
    pos: u32,
    /// RoPE position.
    rope: u32,
    /// Lowest KV position that exists in this layer.
    min_pos: u32,
    token: u32,
}

/// A row of a step in flight.
struct Row {
    slot: Slot,
    c: u32,
    p: u32,
    /// Drafts this row actually proposes.
    k: u32,
    sample: bool,
    params: SamplingParams,
    /// Tokens at positions `c ..`: host tokens, then drafts as they are proposed.
    toks: Vec<u32>,
    /// Chain-level hidden states by `(level, position)` computed in this step.
    levels: HashMap<(usize, u32), Vec<f32>>,
    /// State loaded at step start (levels at `c - 1`).
    state: Vec<Vec<f32>>,
    /// Target logits by position (needed ones, or every one when recording).
    target_logits: HashMap<u32, Vec<f32>>,
    /// Drafter logits by `(depth, slot)`.
    drafter_logits: Vec<(usize, u32, Vec<f32>)>,
    /// Draft distributions.
    q_rows: Vec<Vec<f64>>,
    drafts: Vec<u32>,
    produced: Vec<u32>,
}

impl Row {
    fn token(&self, pos: u32) -> u32 {
        self.toks[(pos - self.c) as usize]
    }

    fn level(&self, level: usize, pos: u32) -> &[f32] {
        if pos + 1 == self.c {
            return &self.state[level];
        }
        self.levels
            .get(&(level, pos))
            .unwrap_or_else(|| panic!("chain level {level} at {pos} not computed"))
    }

    /// Last position whose KV is valid after the step.
    fn last_valid(&self) -> u32 {
        self.p + self.produced.len().saturating_sub(1) as u32
    }
}

/// The CPU reference executor. See the crate documentation.
pub struct CpuExecutor {
    model: Arc<ReferenceModel>,
    spec: ModelSpec,
    cfg: CpuExecutorConfig,
    pools: Vec<Pool>,
    /// Per target layer: `(group, layer within group)`.
    target_kv: Vec<(usize, usize)>,
    drafter_group: Option<usize>,
    /// First full-attention group (holds every position; used to read back prefixes).
    full_group: usize,
    /// `[slot][group][index]`.
    tables: Vec<Vec<Vec<u32>>>,
    /// `[group][block]`: number of `(slot, index)` mappings.
    mapped: Vec<Vec<u32>>,
    states: Vec<SlotState>,
    records: Mutex<Vec<RowRecord>>,
    zero_log: Vec<(u64, u32, u32)>,
    steps: u64,
}

impl std::fmt::Debug for CpuExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExecutor")
            .field("spec", &self.spec)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

fn kind_of(spec: &AttentionSpec) -> AttentionKind {
    match spec.kind {
        ModelAttention::Global => AttentionKind::Full,
        ModelAttention::Sliding { window } => AttentionKind::Sliding {
            window: window as u32,
        },
    }
}

fn layout_of(spec: &AttentionSpec, num_layers: usize, tap_width: usize) -> GroupLayout {
    GroupLayout {
        num_layers,
        num_kv_heads: spec.num_kv_heads,
        head_dim_qk: spec.head_dim_qk,
        head_dim_v: spec.head_dim_v,
        tap_width,
    }
}

impl CpuExecutor {
    /// An executor over `model`. Panics if the configuration does not fit the model (an
    /// MTP depth naming an unloaded layer, target layers of one attention kind with
    /// different KV shapes).
    pub fn new(model: Arc<ReferenceModel>, cfg: CpuExecutorConfig) -> Self {
        let mc = &model.weights.config;
        for &l in &cfg.mtp_depths {
            assert!(
                l < model.weights.mtp.len(),
                "MTP layer {l} is not loaded ({} are)",
                model.weights.mtp.len()
            );
        }
        let depths = cfg.mtp_depths.len();
        let bs = cfg.block_size as usize;
        let nb = cfg.num_blocks as usize;

        // Target groups: one per (attention kind, KV shape), in first-layer order.
        let mut group_keys: Vec<(AttentionKind, usize, usize, usize)> = Vec::new();
        let mut group_layers: Vec<usize> = Vec::new();
        let mut group_spec: Vec<AttentionSpec> = Vec::new();
        let mut target_kv = Vec::new();
        for l in &mc.layers {
            let a = &l.attention;
            let key = (kind_of(a), a.num_kv_heads, a.head_dim_qk, a.head_dim_v);
            let g = match group_keys.iter().position(|k| *k == key) {
                Some(g) => g,
                None => {
                    group_keys.push(key);
                    group_layers.push(0);
                    group_spec.push(a.clone());
                    group_keys.len() - 1
                }
            };
            target_kv.push((g, group_layers[g]));
            group_layers[g] += 1;
        }
        let full_group = group_keys
            .iter()
            .position(|k| k.0 == AttentionKind::Full)
            .expect("a model with at least one global-attention layer");

        let mut kv_groups = Vec::new();
        let mut pools = Vec::new();
        for (g, a) in group_spec.iter().enumerate() {
            let name = match kind_of(a) {
                AttentionKind::Full => "global".to_string(),
                AttentionKind::Sliding { window } => format!("sliding-{window}"),
            };
            kv_groups.push(KvGroupSpec {
                name,
                role: KvRole::Target,
                attention: kind_of(a),
                num_layers: group_layers[g] as u32,
                num_kv_heads: a.num_kv_heads as u32,
                head_dim_qk: a.head_dim_qk as u32,
                head_dim_v: a.head_dim_v as u32,
                num_blocks: cfg.num_blocks,
            });
            pools.push(Pool::new(layout_of(a, group_layers[g], 0), bs, nb));
        }
        let drafter_group = (depths > 0).then(|| {
            let a = &mc.mtp.attention;
            kv_groups.push(KvGroupSpec {
                name: "mtp".into(),
                role: KvRole::Drafter,
                attention: kind_of(a),
                num_layers: depths as u32,
                num_kv_heads: a.num_kv_heads as u32,
                head_dim_qk: a.head_dim_qk as u32,
                head_dim_v: a.head_dim_v as u32,
                num_blocks: cfg.num_blocks,
            });
            pools.push(Pool::new(
                layout_of(a, depths, depths * mc.hidden_size),
                bs,
                nb,
            ));
            pools.len() - 1
        });

        let spec = ModelSpec {
            vocab_size: mc.vocab_size as u32,
            sampleable_vocab_size: cfg.sampleable_vocab_size,
            block_size: cfg.block_size,
            max_model_len: cfg.max_model_len,
            kv_groups,
            max_draft_tokens: depths as u32,
            num_state_slots: cfg.num_state_slots,
            buckets: cfg.buckets.clone(),
        };
        spec.validate().expect("valid model spec");
        let width = spec.max_blocks_per_seq() as usize;
        let groups = spec.kv_groups.len();
        Self {
            tables: vec![vec![vec![NULL_BLOCK; width]; groups]; cfg.num_state_slots as usize],
            mapped: vec![vec![0; nb]; groups],
            states: vec![SlotState::zeroed(depths, mc.hidden_size); cfg.num_state_slots as usize],
            records: Mutex::new(Vec::new()),
            zero_log: Vec::new(),
            steps: 0,
            model,
            spec,
            cfg,
            pools,
            target_kv,
            drafter_group,
            full_group,
        }
    }

    /// The model.
    pub fn model(&self) -> &Arc<ReferenceModel> {
        &self.model
    }

    /// The configuration.
    pub fn config(&self) -> &CpuExecutorConfig {
        &self.cfg
    }

    /// Steps executed.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Every executed zero as `(step, group, block)`.
    pub fn zero_log(&self) -> &[(u64, u32, u32)] {
        &self.zero_log
    }

    /// Whether every byte of a block (KV, drafter taps and tags) is zero.
    pub fn block_is_zero(&self, group: u32, block: u32) -> bool {
        self.pools[group as usize].block_is_zero(block)
    }

    /// Whether a slot's per-sequence model state (the drafter's hidden states) is
    /// scrubbed: full-size buffers holding only zero bytes, at no position.
    pub fn slot_state_is_zero(&self, slot: Slot) -> bool {
        let hsz = self.model.weights.config.hidden_size;
        let st = &self.states[slot as usize];
        st.at.is_none()
            && st.levels.len() == self.levels()
            && st
                .levels
                .iter()
                .all(|l| l.len() == hsz && l.iter().all(|x| x.to_bits() == 0))
    }

    /// Records kept since the last call (with [`CpuExecutorConfig::record`]). Takes
    /// `&self` so a caller that only borrows the executor (through the engine) can drain
    /// them.
    pub fn take_records(&self) -> Vec<RowRecord> {
        std::mem::take(&mut *self.records.lock().expect("records lock"))
    }

    fn levels(&self) -> usize {
        self.cfg.mtp_depths.len()
    }

    fn set_table(&mut self, slot: Slot, group: u32, index: u32, block: u32) {
        let (g, s) = (group as usize, slot as usize);
        assert!(
            (block as usize) < self.pools[g].num_blocks(),
            "table update to block {block} outside group {group}"
        );
        let cur = &mut self.tables[s][g][index as usize];
        let old = std::mem::replace(cur, block);
        if old != NULL_BLOCK {
            self.mapped[g][old as usize] -= 1;
        }
        if block != NULL_BLOCK {
            self.mapped[g][block as usize] += 1;
        }
    }

    fn reset_slot(&mut self, slot: Slot) {
        for g in 0..self.spec.kv_groups.len() {
            for idx in 0..self.tables[slot as usize][g].len() {
                if self.tables[slot as usize][g][idx] != NULL_BLOCK {
                    self.set_table(slot, g as u32, idx as u32, NULL_BLOCK);
                }
            }
        }
        self.states[slot as usize].scrub();
    }

    fn block_of(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        let block = self.tables[slot as usize][group][(pos / self.cfg.block_size) as usize];
        assert!(
            block != NULL_BLOCK,
            "slot {slot} group {group} position {pos}: unmapped block"
        );
        block
    }

    /// The block a write at `pos` lands in; it must be mapped by exactly this one table
    /// entry (blocks shared through the prefix cache are immutable).
    fn writable_block(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        let block = self.block_of(slot, group, pos);
        assert!(
            self.mapped[group][block as usize] == 1,
            "slot {slot} group {group} position {pos}: write to shared block {block}"
        );
        block
    }

    /// Reads back the token stored at every position `0..n` through the slot's table.
    fn prefix_tokens(&self, slot: Slot, n: u32) -> Vec<u32> {
        let bs = self.cfg.block_size;
        (0..n)
            .map(|pos| {
                let block = self.block_of(slot, self.full_group, pos);
                let tag = self.pools[self.full_group].tag(block, 0, (pos % bs) as usize);
                assert_eq!(
                    tag.pos_plus_one,
                    pos + 1,
                    "slot {slot}: position {pos} holds foreign or zeroed KV"
                );
                tag.token
            })
            .collect()
    }

    /// Attention over paged KV for `x`'s rows: projection, RoPE, KV write through the
    /// tables, attention over KV read through the tables, `o_proj`. Mirrors
    /// [`ReferenceModel::attention`] operation for operation.
    fn paged_attention(
        &mut self,
        group: usize,
        layer: usize,
        spec: &AttentionSpec,
        aw: &AttentionWeights,
        x: &Matrix,
        rows: &[AttnRow],
    ) -> Matrix {
        let (dq, dv) = (spec.head_dim_qk, spec.head_dim_v);
        let (nq, nkv) = (spec.num_q_heads, spec.num_kv_heads);
        let qkv = linear(x, &aw.qkv);
        let v_scale = if self.model.weights.value_scale_folded {
            None
        } else {
            self.model.weights.config.attention_value_scale
        };
        let bs = self.cfg.block_size;
        let mut q = Matrix::zeros(rows.len(), nq * dq);
        for (i, r) in rows.iter().enumerate() {
            let row = qkv.row(i);
            q.row_mut(i).copy_from_slice(&row[..nq * dq]);
            let mut k = row[nq * dq..(nq + nkv) * dq].to_vec();
            let mut v = row[(nq + nkv) * dq..].to_vec();
            if let Some(s) = v_scale {
                for x in &mut v {
                    *x *= s;
                }
            }
            let (cos, sin) = rope_cos_sin(spec, r.rope as usize);
            for hq in q.row_mut(i).chunks_exact_mut(dq) {
                apply_rope(hq, &cos, &sin);
            }
            for hk in k.chunks_exact_mut(dq) {
                apply_rope(hk, &cos, &sin);
            }
            let block = self.writable_block(r.slot, group, r.pos);
            let tag = Tag {
                pos_plus_one: r.pos + 1,
                token: r.token,
            };
            self.pools[group].write(block, layer, (r.pos % bs) as usize, tag, &k, &v);
        }

        let group_size = spec.group_size();
        let pool = &self.pools[group];
        let tables = &self.tables;
        let mut out = Matrix::zeros(x.rows, nq * dv);
        out.data
            .par_chunks_mut(nq * dv)
            .zip(rows.par_iter())
            .enumerate()
            .for_each(|(i, (orow, r))| {
                let lo = match spec.window() {
                    Some(w) => (r.pos + 1).saturating_sub(w as u32),
                    None => 0,
                }
                .max(r.min_pos);
                let kv: Vec<(&[f32], &[f32])> = (lo..=r.pos)
                    .map(|pos| {
                        let block = tables[r.slot as usize][group][(pos / bs) as usize];
                        assert!(
                            block != NULL_BLOCK,
                            "slot {} group {group} position {pos}: read of an unmapped block",
                            r.slot
                        );
                        let (k, v, tag) = pool.read(block, layer, (pos % bs) as usize);
                        assert_eq!(
                            tag.pos_plus_one,
                            pos + 1,
                            "slot {} group {group} layer {layer} position {pos}: read of \
                             zeroed or foreign KV",
                            r.slot
                        );
                        (k, v)
                    })
                    .collect();
                for head in 0..nq {
                    let g = head / group_size;
                    let keys: Vec<&[f32]> =
                        kv.iter().map(|(k, _)| &k[g * dq..(g + 1) * dq]).collect();
                    let values: Vec<&[f32]> =
                        kv.iter().map(|(_, v)| &v[g * dv..(g + 1) * dv]).collect();
                    let sink = aw.sinks.as_ref().map(|s| s[head]);
                    attend(
                        spec,
                        &q.row(i)[head * dq..(head + 1) * dq],
                        &keys,
                        &values,
                        sink,
                        &mut orow[head * dv..(head + 1) * dv],
                    );
                }
            });
        linear(&out, &aw.o_proj)
    }

    fn embed(
        &self,
        tokens: impl Iterator<Item = u32>,
        rows: usize,
    ) -> Result<Matrix, ExecutorError> {
        let w = &self.model.weights;
        let mut x = Matrix::zeros(rows, w.config.hidden_size);
        for (i, t) in tokens.enumerate() {
            if t as usize >= w.config.vocab_size {
                return Err(ExecutorError(format!(
                    "token {t} outside vocab {}",
                    w.config.vocab_size
                )));
            }
            x.row_mut(i).copy_from_slice(w.embed.row(t as usize));
        }
        Ok(x)
    }

    /// The main model over `rows` (real rows first; the matrix may carry padding rows
    /// after them). Returns the last layer's output and its final norm.
    fn target_forward(
        &mut self,
        rows: &[AttnRow],
        padded: usize,
    ) -> Result<(Matrix, Matrix), ExecutorError> {
        let model = self.model.clone();
        let w = &model.weights;
        let eps = w.config.rms_norm_eps;
        let mut h = self.embed(rows.iter().map(|r| r.token), padded.max(rows.len()))?;
        for (layer, lw) in w.layers.iter().enumerate() {
            let spec = &w.config.layers[layer].attention;
            let (group, gl) = self.target_kv[layer];
            let x = rms_norm_rows(&h, &lw.input_norm, eps);
            let a = self.paged_attention(group, gl, spec, &lw.attention, &x, rows);
            add_assign(&mut h, &a);
            let x = rms_norm_rows(&h, &lw.post_attention_norm, eps);
            let f = match &lw.ffn {
                FfnWeights::Dense(d) => model.dense_ffn(d, &x),
                FfnWeights::Moe(_) => model
                    .moe(layer, &x)
                    .map_err(|e| ExecutorError(e.to_string()))?,
            };
            add_assign(&mut h, &f);
        }
        let normed = rms_norm_rows(&h, &w.final_norm, eps);
        Ok((h, normed))
    }

    /// MTP depth `depth` over `rows`, each continuing from the hidden state in `prev`'s
    /// matching row. Mirrors [`ReferenceModel::mtp_forward`].
    fn mtp_forward(
        &mut self,
        depth: usize,
        rows: &[AttnRow],
        prev: &Matrix,
    ) -> Result<(Matrix, Matrix), ExecutorError> {
        let model = self.model.clone();
        let w = &model.weights;
        let mw = &w.mtp[self.cfg.mtp_depths[depth]];
        let cfg = &w.config;
        let hsz = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let e = rms_norm_rows(
            &self.embed(rows.iter().map(|r| r.token), rows.len())?,
            &mw.enorm,
            eps,
        );
        let hp = rms_norm_rows(prev, &mw.hnorm, eps);
        let mut cat = Matrix::zeros(rows.len(), 2 * hsz);
        for i in 0..rows.len() {
            let r = cat.row_mut(i);
            r[..hsz].copy_from_slice(e.row(i));
            r[hsz..].copy_from_slice(hp.row(i));
        }
        let mut h = linear(&cat, &mw.eh_proj);
        let x = rms_norm_rows(&h, &mw.input_norm, eps);
        let group = self.drafter_group.expect("drafter group");
        let a = self.paged_attention(group, depth, &cfg.mtp.attention, &mw.attention, &x, rows);
        add_assign(&mut h, &a);
        let x = rms_norm_rows(&h, &mw.post_attention_norm, eps);
        let f = model.dense_ffn(&mw.ffn, &x);
        add_assign(&mut h, &f);
        let normed = rms_norm_rows(&h, &mw.final_norm, eps);
        Ok((h, normed))
    }

    fn lm_head(&self, normed: &Matrix, rows: &[usize]) -> Matrix {
        linear(&normed.select_rows(rows), &self.model.weights.lm_head)
    }

    fn chain_state<'a>(&self, hidden: &'a Matrix, normed: &'a Matrix, i: usize) -> &'a [f32] {
        match self.cfg.mtp_hidden {
            MtpHidden::Normed => normed.row(i),
            MtpHidden::PreNorm => hidden.row(i),
        }
    }

    /// Runs the target over `(row, position)` pairs (tokens taken from the rows), storing
    /// chain level 0 and the logits each row needs.
    fn run_target(
        &mut self,
        rows: &mut [Row],
        items: &[(usize, u32)],
        padded: usize,
        need_logits: impl Fn(&Row, u32) -> bool,
    ) -> Result<(), ExecutorError> {
        if items.is_empty() {
            return Ok(());
        }
        let attn: Vec<AttnRow> = items
            .iter()
            .map(|&(ri, pos)| AttnRow {
                slot: rows[ri].slot,
                pos,
                rope: pos,
                min_pos: 0,
                token: rows[ri].token(pos),
            })
            .collect();
        let (hidden, normed) = self.target_forward(&attn, padded)?;
        let chained = self.levels() > 0;
        let mut want = Vec::new();
        for (i, &(ri, pos)) in items.iter().enumerate() {
            if chained {
                let s = self.chain_state(&hidden, &normed, i).to_vec();
                rows[ri].levels.insert((0, pos), s);
            }
            if self.cfg.record || need_logits(&rows[ri], pos) {
                want.push(i);
            }
        }
        if !want.is_empty() {
            let logits = self.lm_head(&normed, &want);
            for (j, &i) in want.iter().enumerate() {
                let (ri, pos) = items[i];
                rows[ri].target_logits.insert(pos, logits.row(j).to_vec());
            }
        }
        Ok(())
    }

    /// Runs MTP depth `depth` over `(row, slot position)` pairs. A row at slot `s` consumes
    /// the token at `s` and chain level `depth` at `s - 1`, uses RoPE position `s - 1`, and
    /// writes its KV at `s`. Stores chain level `depth + 1` and returns the logits of the
    /// items listed in `logits_for` (plus every item's when recording).
    fn run_depth(
        &mut self,
        rows: &mut [Row],
        depth: usize,
        items: &[(usize, u32)],
        logits_for: &[(usize, u32)],
    ) -> Result<HashMap<(usize, u32), Vec<f32>>, ExecutorError> {
        let mut out = HashMap::new();
        if items.is_empty() {
            return Ok(out);
        }
        let hsz = self.model.weights.config.hidden_size;
        let mut prev = Matrix::zeros(items.len(), hsz);
        let attn: Vec<AttnRow> = items
            .iter()
            .enumerate()
            .map(|(i, &(ri, s))| {
                prev.row_mut(i)
                    .copy_from_slice(rows[ri].level(depth, s - 1));
                AttnRow {
                    slot: rows[ri].slot,
                    pos: s,
                    rope: s - 1,
                    min_pos: depth as u32 + 1,
                    token: rows[ri].token(s),
                }
            })
            .collect();
        let (hidden, normed) = self.mtp_forward(depth, &attn, &prev)?;
        let mut want = Vec::new();
        for (i, &(ri, s)) in items.iter().enumerate() {
            let state = self.chain_state(&hidden, &normed, i).to_vec();
            rows[ri].levels.insert((depth + 1, s), state);
            if self.cfg.record || logits_for.contains(&(ri, s)) {
                want.push(i);
            }
        }
        if !want.is_empty() {
            let logits = self.lm_head(&normed, &want);
            for (j, &i) in want.iter().enumerate() {
                let (ri, s) = items[i];
                let l = logits.row(j).to_vec();
                if self.cfg.record {
                    rows[ri].drafter_logits.push((depth, s, l.clone()));
                }
                if logits_for.contains(&(ri, s)) {
                    out.insert((ri, s), l);
                }
            }
        }
        Ok(out)
    }

    /// Loads the drafter state a row continues from: the slot's own state when it is at
    /// `c - 1`, else the boundary tap of the block ending at `c - 1` (a prefix-cache hit
    /// or a resume always starts on a block boundary).
    fn load_state(&self, e: &SeqEntry) -> Vec<Vec<f32>> {
        let levels = self.levels();
        if levels == 0 || e.context_len == 0 {
            return Vec::new();
        }
        let at = e.context_len - 1;
        let st = &self.states[e.slot as usize];
        if st.at == Some(at) {
            return st.levels.clone();
        }
        let g = self.drafter_group.expect("drafter group");
        let block = self.block_of(e.slot, g, at);
        let tap = self.pools[g].read_tap(block, at).unwrap_or_else(|| {
            panic!(
                "slot {}: no drafter state at position {at} (not a block boundary of a \
                 cached prefix, and not this slot's last position)",
                e.slot
            )
        });
        let hsz = self.model.weights.config.hidden_size;
        tap.chunks_exact(hsz).map(|c| c.to_vec()).collect()
    }

    /// Writes boundary taps for the row's block-final positions in `from ..= to`, and
    /// leaves the slot's state at `to`.
    fn store_state(&mut self, row: &Row, from: u32, to: u32) {
        let levels = self.levels();
        if levels == 0 {
            return;
        }
        let hsz = self.model.weights.config.hidden_size;
        let level_at = |l: usize, pos: u32| -> Vec<f32> {
            // Level `l > 0` at `pos` exists only from position `l` on.
            if l > 0 && pos < l as u32 {
                return vec![0.0; hsz];
            }
            row.level(l, pos).to_vec()
        };
        let g = self.drafter_group.expect("drafter group");
        let bs = self.cfg.block_size;
        for pos in from..=to {
            if (pos + 1) % bs == 0 {
                let tap: Vec<f32> = (0..levels).flat_map(|l| level_at(l, pos)).collect();
                let block = self.writable_block(row.slot, g, pos);
                self.pools[g].write_tap(block, pos, &tap);
            }
        }
        let state = &mut self.states[row.slot as usize];
        for (l, buf) in state.levels.iter_mut().enumerate() {
            buf.copy_from_slice(&level_at(l, to));
        }
        state.at = Some(to);
    }
}

impl Executor for CpuExecutor {
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
        for m in &step.maintenance {
            match *m {
                Maintenance::Zero { group, block } => {
                    self.pools[group as usize].zero(block);
                    self.zero_log.push((self.steps, group, block));
                }
                Maintenance::Copy { group, src, dst } => {
                    self.pools[group as usize].copy(src, dst);
                }
                Maintenance::ResetSlot { slot } => self.reset_slot(slot),
            }
        }
        for u in &step.table_updates {
            self.set_table(u.slot, u.group, u.index, u.block);
        }

        let depths = self.levels();
        let mut rows: Vec<Row> = Vec::with_capacity(step.seqs.len());
        for e in &step.seqs {
            assert!(e.num_tokens >= 1, "a row needs at least one host token");
            let start = e.token_start as usize;
            let end = start + e.num_tokens as usize;
            for (i, &pos) in step.positions[start..end].iter().enumerate() {
                assert_eq!(
                    pos,
                    e.context_len + i as u32,
                    "positions disagree with context_len"
                );
            }
            let p = e.context_len + e.num_tokens - 1;
            let k = if e.sample { e.num_drafts } else { 0 };
            assert!(
                k as usize <= depths,
                "{k} drafts requested; the drafter has {depths} depths"
            );
            // No drafter row exists for position 0 (MTP continues from the previous
            // position's hidden state), and the seam forbids asking for one.
            assert!(
                p > 0 || k == 0,
                "slot {}: {k} drafts requested at position 0",
                e.slot
            );
            rows.push(Row {
                slot: e.slot,
                c: e.context_len,
                p,
                k,
                sample: e.sample,
                params: e.sampling,
                toks: step.token_ids[start..end].to_vec(),
                levels: HashMap::new(),
                state: self.load_state(e),
                target_logits: HashMap::new(),
                drafter_logits: Vec::new(),
                q_rows: Vec::new(),
                drafts: Vec::new(),
                produced: Vec::new(),
            });
        }

        // 1. Target over every host token.
        let host: Vec<(usize, u32)> = rows
            .iter()
            .enumerate()
            .flat_map(|(ri, r)| (r.c..=r.p).map(move |pos| (ri, pos)))
            .collect();
        let drafted: u32 = rows.iter().map(|r| r.k).sum();
        let padded = if self.cfg.pad_batches {
            (step.bucket.max_tokens - drafted) as usize
        } else {
            host.len()
        };
        self.run_target(&mut rows, &host, padded, |r, pos| r.sample && pos == r.p)?;

        // 2. Drafter over every host position, depth by depth: depth `d` at slot `s`
        //    consumes the token at `s` and depth `d - 1`'s state at `s - 1`.
        let mut first_draft_logits = HashMap::new();
        for d in 0..depths {
            let items: Vec<(usize, u32)> = rows
                .iter()
                .enumerate()
                .flat_map(|(ri, r)| (r.c.max(d as u32 + 1)..=r.p).map(move |s| (ri, s)))
                .collect();
            let wanted: Vec<(usize, u32)> = if d == 0 {
                rows.iter()
                    .enumerate()
                    .filter(|(_, r)| r.k > 0)
                    .map(|(ri, r)| (ri, r.p))
                    .collect()
            } else {
                Vec::new()
            };
            let l = self.run_depth(&mut rows, d, &items, &wanted)?;
            if d == 0 {
                first_draft_logits = l;
            }
        }

        // 3. The draft chain: draft `i + 1` comes from depth `i` at slot `p + i`. Depth
        //    `i` first runs over slots `p + 1 ..= p + i`, consuming drafts `1 ..= i`.
        let (vocab, sampleable) = (self.spec.vocab_size, self.spec.sampleable_vocab_size);
        let draft_from = |row: &mut Row, logits: &[f32]| {
            assert_eq!(
                logits.len(),
                vocab as usize,
                "a drafter logit row is a full row"
            );
            let logits = Logits::new(logits, sampleable);
            let pos = row.p as u64 + 1 + row.drafts.len() as u64;
            let (d, q) = if row.params.is_greedy() {
                let d = sampling::argmax(logits);
                let mut q = vec![0.0; logits.len()];
                q[d as usize] = 1.0;
                (d, q)
            } else {
                let q = sampling::processed_probs(logits, &row.params);
                let d = sampling::sample_from(
                    &q,
                    sampling::uniform(row.params.seed(), pos, Stream::Draft),
                );
                (d, q)
            };
            row.drafts.push(d);
            row.toks.push(d);
            row.q_rows.push(q);
        };
        for (ri, row) in rows.iter_mut().enumerate() {
            if row.k > 0 {
                draft_from(row, &first_draft_logits[&(ri, row.p)]);
            }
        }
        for i in 1..depths {
            let items: Vec<(usize, u32)> = rows
                .iter()
                .enumerate()
                .filter(|(_, r)| r.k as usize > i)
                .flat_map(|(ri, r)| {
                    ((r.p + 1).max(i as u32 + 1)..=r.p + i as u32).map(move |s| (ri, s))
                })
                .collect();
            let wanted: Vec<(usize, u32)> = rows
                .iter()
                .enumerate()
                .filter(|(_, r)| r.k as usize > i)
                .map(|(ri, r)| (ri, r.p + i as u32))
                .collect();
            let logits = self.run_depth(&mut rows, i, &items, &wanted)?;
            for (ri, s) in wanted {
                draft_from(&mut rows[ri], &logits[&(ri, s)]);
            }
        }

        // 4. Target over the drafts (verification).
        let verify: Vec<(usize, u32)> = rows
            .iter()
            .enumerate()
            .flat_map(|(ri, r)| (r.p + 1..=r.p + r.k).map(move |pos| (ri, pos)))
            .collect();
        self.run_target(&mut rows, &verify, verify.len(), |_, _| true)?;

        // 5. Sampling and chain acceptance.
        for row in &mut rows {
            if !row.sample {
                continue;
            }
            let params = row.params;
            row.produced = if row.k == 0 {
                vec![sampling::sample(
                    self.spec.logits(&row.target_logits[&row.p]),
                    &params,
                    row.p as u64 + 1,
                )]
            } else {
                let p_rows: Vec<Vec<f64>> = (row.p..=row.p + row.k)
                    .map(|pos| {
                        sampling::processed_probs(
                            self.spec.logits(&row.target_logits[&pos]),
                            &params,
                        )
                    })
                    .collect();
                sampling::chain_accept(&p_rows, &row.q_rows, &row.drafts, &params, row.p as u64 + 1)
            };
        }

        // 6. Drafter rows for accepted drafts the chain did not already compute (depth `d`
        //    ran over slots `p + 1 ..= p + d` while drafting; those rows consumed only
        //    accepted tokens, so they are exact).
        for d in 0..depths {
            let items: Vec<(usize, u32)> = rows
                .iter()
                .enumerate()
                .flat_map(|(ri, r)| {
                    let done = if d >= 1 && d < r.k as usize {
                        d as u32
                    } else {
                        0
                    };
                    let lo = (r.p + 1 + done).max(d as u32 + 1);
                    (lo..=r.last_valid()).map(move |s| (ri, s))
                })
                .collect();
            self.run_depth(&mut rows, d, &items, &[])?;
        }

        // 7. Drafter state and boundary taps for every valid position.
        for row in &rows {
            self.store_state(row, row.c, row.last_valid());
        }

        // Output.
        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; rows.len() * stride as usize],
            stride,
            num_tokens: vec![0; rows.len()],
            logits: step.return_logits.then(Vec::new),
        };
        for (i, row) in rows.iter().enumerate() {
            let base = i * stride as usize;
            out.tokens[base..base + row.produced.len()].copy_from_slice(&row.produced);
            out.num_tokens[i] = row.produced.len() as u32;
            if let Some(l) = out.logits.as_mut() {
                l.push(
                    (0..row.produced.len() as u32)
                        .map(|j| row.target_logits[&(row.p + j)].clone())
                        .collect(),
                );
            }
        }
        if self.cfg.record {
            for row in rows {
                let mut tokens = self.prefix_tokens(row.slot, row.c);
                tokens.extend(&row.toks);
                let mut target_logits: Vec<(u32, Vec<f32>)> =
                    row.target_logits.into_iter().collect();
                target_logits.sort_by_key(|(p, _)| *p);
                self.records
                    .get_mut()
                    .expect("records lock")
                    .push(RowRecord {
                        slot: row.slot,
                        context_len: row.c,
                        num_tokens: row.p + 1 - row.c,
                        tokens,
                        target_logits,
                        drafter_logits: row.drafter_logits,
                        drafts: row.drafts,
                        produced: row.produced,
                        sampling: row.params,
                    });
            }
        }
        Ok(out)
    }
}
