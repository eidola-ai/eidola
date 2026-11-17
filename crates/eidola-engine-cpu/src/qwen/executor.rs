//! [`QwenCpuExecutor`]: the serving core's [`Executor`] for Qwen3.8 over the model crate's
//! reference numerics.
//!
//! Every computation is the model crate's: the DeltaNet's per-token core
//! (`deltanet::token_core`), its projections (`linear::lin8_rows`,
//! `linear::linear_bf16_dense`), the norms and residual adds, the attention layer
//! (`QwenModel::layer_forward` over the KV gathered from the pages), the MLP, the head
//! and the drafter (`QwenModel::mtp_forward`). The executor only decides what runs where,
//! and keeps what the seam says it keeps.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use eidola_engine::executor::{
    Executor, ExecutorError, Maintenance, SeqEntry, Slot, StateRow, StateWrite, StepInput,
    StepOutput,
};
use eidola_engine::sampling::{self, Logits, SamplingParams, Stream};
use eidola_engine::spec::{
    AttentionKind, Bucket, Coverage, KvGroupSpec, KvRole, ModelSpec, NULL_BLOCK, NULL_STATE,
    StateGroupSpec, StateSpec,
};
use eidola_engine_model::Matrix;
use eidola_engine_model::qwen::config::LayerKind;
use eidola_engine_model::qwen::deltanet::{CoreWeights, token_core};
use eidola_engine_model::qwen::forward::{residual_add, rms_norm_zc_rows};
use eidola_engine_model::qwen::linear::{lin8_rows, linear_bf16_dense};
use eidola_engine_model::qwen::weights::{LayerWeights, MixerWeights};
use eidola_engine_model::qwen::{DeltaNetState, KvCache, LayerState, LogitRows, QwenModel};

use crate::page::{PageBuf, Tag};
use crate::pool::{GroupLayout, Pool};
use crate::qwen::state::{StatePool, TailEntry, layer_floats, tail_floats_per_layer};

/// KV group of the target's attention layers.
const TARGET: usize = 0;
/// KV group of the drafter's layer, when drafting.
const DRAFTER: usize = 1;

/// Executor configuration: memory geometry and drafting. The model fixes the rest.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenCpuExecutorConfig {
    /// Positions per KV block, and the boundaries the recurrent state is stored at.
    pub block_size: u32,
    /// Physical blocks per KV group, including the null block 0.
    pub num_blocks: u32,
    /// Host-tier blocks per KV group, including the null block 0; 0 for no host tier
    /// (then `host_state_buffers` is 0 too). Host pools are locked into memory, and the
    /// executor refuses to start (panics) if that fails.
    pub host_blocks: u32,
    /// State buffers, including the reserved zero state 0.
    pub state_buffers: u32,
    /// Host-tier state buffers, including a reserved buffer 0.
    pub host_state_buffers: u32,
    /// Per-sequence state rows.
    pub num_state_slots: u32,
    /// Longest sequence.
    pub max_model_len: u32,
    /// Captured step shapes, ascending.
    pub buckets: Vec<Bucket>,
    /// Drafted tokens per step (`k`), all from the model's one MTP layer recursing on its
    /// own output; 0 disables drafting and the drafter's KV group.
    pub draft_tokens: u32,
    /// Token ids the model may emit (the tokenizer's vocabulary); the head's rows past it
    /// take no part in sampling, drafting or acceptance.
    pub sampleable_vocab_size: u32,
    /// Testing aid: compute logits at every computed position and keep a
    /// [`QwenRowRecord`] of every row and every stored state.
    pub record: bool,
}

/// Everything one row of one step computed, for checking against the dense oracle.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenRowRecord {
    pub slot: Slot,
    /// Positions already in KV before the step.
    pub context_len: u32,
    pub num_tokens: u32,
    /// Tokens at positions `0 ..= p + k`: the prefix read back through the block tables,
    /// the host tokens, then the drafts.
    pub tokens: Vec<u32>,
    /// Target logits by position: every host position and every drafted one.
    pub target_logits: Vec<(u32, Vec<f32>)>,
    /// The drafter's canonical rows' logits by position (host positions, and accepted
    /// drafts' rows rewritten from the target's states).
    pub drafter_logits: Vec<(u32, Vec<f32>)>,
    /// The draft chain's logits, one row per draft: the logits each was drawn from.
    pub chain_logits: Vec<Vec<f32>>,
    pub drafts: Vec<u32>,
    pub produced: Vec<u32>,
    pub sampling: SamplingParams,
}

/// A state the executor stored (with [`QwenCpuExecutorConfig::record`]): the boundary
/// and the layers' states there, as stored.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredState {
    pub slot: Slot,
    pub len: u32,
    /// The tokens before `len`, read back through the slot's block tables.
    pub tokens: Vec<u32>,
    pub states: Vec<DeltaNetState>,
}

/// Per-slot model state: the target's normed final states at the positions the slot's
/// last row ended on (its last host token and its drafts), which the drafter's next row
/// continues from, each with its position plus one. Allocated once and overwritten in
/// place; `ResetSlot` zeroes it.
struct Hiddens {
    data: PageBuf<f32>,
    tags: PageBuf<u32>,
    width: usize,
}

impl Hiddens {
    fn new(rows: usize, width: usize) -> Self {
        Self {
            data: PageBuf::zeroed(rows * width),
            tags: PageBuf::zeroed(rows),
            width,
        }
    }

    fn set(&mut self, rows: &[(u32, Vec<f32>)]) {
        self.data.fill(0.0);
        self.tags.fill(0);
        for (i, (pos, h)) in rows.iter().enumerate() {
            self.data[i * self.width..(i + 1) * self.width].copy_from_slice(h);
            self.tags[i] = pos + 1;
        }
    }

    fn get(&self, pos: u32) -> Option<Vec<f32>> {
        let i = self.tags.iter().position(|&t| t == pos + 1)?;
        Some(self.data[i * self.width..(i + 1) * self.width].to_vec())
    }

    fn reset(&mut self) {
        self.data.fill(0.0);
        self.tags.fill(0);
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.to_bits() == 0) && self.tags.iter().all(|&t| t == 0)
    }
}

/// The CPU executor for Qwen3.8.
pub struct QwenCpuExecutor {
    model: Arc<QwenModel>,
    cfg: QwenCpuExecutorConfig,
    spec: ModelSpec,
    /// Model layer indices of the DeltaNet layers.
    deltanet: Vec<usize>,
    pools: Vec<Pool>,
    host_pools: Vec<Pool>,
    states: StatePool,
    host_states: Option<StatePool>,
    /// `[slot][group][logical block]`.
    tables: Vec<Vec<Vec<u32>>>,
    tails: Vec<crate::qwen::state::SlotTail>,
    hiddens: Vec<Hiddens>,
    steps: u64,
    zero_log: Vec<(u64, u32, u32)>,
    crypt_log: Vec<(u64, u32, u32, bool)>,
    transfer_log: Vec<(u64, u32, u32, u32, bool)>,
    host_zero_log: Vec<(u64, u32, u32)>,
    records: Mutex<Vec<QwenRowRecord>>,
    stored: Mutex<Vec<StoredState>>,
}

impl std::fmt::Debug for QwenCpuExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenCpuExecutor")
            .field("cfg", &self.cfg)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

/// What one row carries through a step.
struct Row {
    slot: Slot,
    c: u32,
    /// The last host position.
    p: u32,
    k: u32,
    sample: bool,
    params: SamplingParams,
    toks: Vec<u32>,
    base: StateRow,
    writes: Vec<StateWrite>,
    drafts: Vec<u32>,
    q_rows: Vec<Vec<f64>>,
    produced: Vec<u32>,
    target_logits: BTreeMap<u32, Vec<f32>>,
    drafter_logits: Vec<(u32, Vec<f32>)>,
    chain_logits: Vec<Vec<f32>>,
}

impl QwenCpuExecutor {
    /// An executor over `model`. Panics on a configuration the model cannot run.
    pub fn new(model: Arc<QwenModel>, cfg: QwenCpuExecutorConfig) -> Self {
        let c = model.config().clone();
        let deltanet: Vec<usize> = c.layers_of(LayerKind::DeltaNet).collect();
        let attention: Vec<usize> = c.layers_of(LayerKind::Attention).collect();
        let a = &c.attention;
        let k = cfg.draft_tokens;
        assert!(
            k == 0 || !model.weights.mtp.is_empty(),
            "drafting needs the model's MTP layer"
        );
        let layout = |layers: usize, tap: usize| GroupLayout {
            num_layers: layers,
            num_kv_heads: a.num_kv_heads,
            head_dim_qk: a.head_dim,
            head_dim_v: a.head_dim,
            tap_width: tap,
        };
        let mut layouts = vec![layout(attention.len(), 0)];
        if k > 0 {
            layouts.push(layout(1, c.hidden_size));
        }
        let bs = cfg.block_size as usize;
        let pools: Vec<Pool> = layouts
            .iter()
            .map(|l| Pool::new(l.clone(), bs, cfg.num_blocks as usize))
            .collect();
        let tiered = cfg.host_blocks > 0;
        assert_eq!(
            tiered,
            cfg.host_state_buffers > 0,
            "a host tier for the KV and the state alike, or for neither"
        );
        let host_pools: Vec<Pool> = if tiered {
            layouts
                .iter()
                .map(|l| {
                    let mut p = Pool::new(l.clone(), bs, cfg.host_blocks as usize);
                    p.lock().unwrap_or_else(|e| refuse_lock(e));
                    p
                })
                .collect()
        } else {
            Vec::new()
        };
        let states = StatePool::new(&c.deltanet, deltanet.len(), cfg.state_buffers as usize);
        let host_states = tiered.then(|| {
            let mut p =
                StatePool::new(&c.deltanet, deltanet.len(), cfg.host_state_buffers as usize);
            p.lock().unwrap_or_else(|e| refuse_lock(e));
            p
        });
        let group = |name: &str, role, layers: usize| KvGroupSpec {
            name: name.into(),
            role,
            attention: AttentionKind::Full,
            num_layers: layers as u32,
            num_kv_heads: a.num_kv_heads as u32,
            head_dim_qk: a.head_dim as u32,
            head_dim_v: a.head_dim as u32,
            num_blocks: cfg.num_blocks,
            host_blocks: cfg.host_blocks,
            ratio: 1,
            device: 0,
            coverage: Coverage::All,
        };
        let mut kv_groups = vec![group("attention", KvRole::Target, attention.len())];
        if k > 0 {
            kv_groups.push(group("mtp", KvRole::Drafter, 1));
        }
        let tail_positions = cfg.block_size - 1 + k;
        let spec = ModelSpec {
            vocab_size: c.vocab_size as u32,
            sampleable_vocab_size: cfg.sampleable_vocab_size,
            block_size: cfg.block_size,
            max_model_len: cfg.max_model_len,
            kv_groups,
            max_draft_tokens: k,
            num_state_slots: cfg.num_state_slots,
            buckets: cfg.buckets.clone(),
            window_chain: None,
            token_history: 0,
            state: Some(StateSpec {
                groups: vec![StateGroupSpec {
                    name: "deltanet".into(),
                    num_layers: deltanet.len() as u32,
                    device: 0,
                    image_bytes: (deltanet.len() * layer_floats(&c.deltanet) * 4 + 4) as u64,
                }],
                num_buffers: cfg.state_buffers,
                host_buffers: cfg.host_state_buffers,
                tail_positions,
            }),
        };
        spec.validate().expect("a valid spec");
        let slots = cfg.num_state_slots as usize;
        let width = spec.max_blocks_per_seq() as usize;
        Self {
            tables: vec![vec![vec![NULL_BLOCK; width]; layouts.len()]; slots],
            tails: (0..slots)
                .map(|_| {
                    crate::qwen::state::SlotTail::new(
                        &c.deltanet,
                        deltanet.len(),
                        tail_positions as usize,
                    )
                })
                .collect(),
            hiddens: (0..slots)
                .map(|_| Hiddens::new(k as usize + 1, c.hidden_size))
                .collect(),
            model,
            cfg,
            spec,
            deltanet,
            pools,
            host_pools,
            states,
            host_states,
            steps: 0,
            zero_log: Vec::new(),
            crypt_log: Vec::new(),
            transfer_log: Vec::new(),
            host_zero_log: Vec::new(),
            records: Mutex::new(Vec::new()),
            stored: Mutex::new(Vec::new()),
        }
    }

    pub fn model(&self) -> &Arc<QwenModel> {
        &self.model
    }

    pub fn config(&self) -> &QwenCpuExecutorConfig {
        &self.cfg
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The pool index maintenance names the state pool by.
    fn state_pool(&self) -> u32 {
        self.pools.len() as u32
    }

    /// Every executed zero as `(step, group, block)` (the state pool's index for a state
    /// buffer).
    pub fn zero_log(&self) -> &[(u64, u32, u32)] {
        &self.zero_log
    }

    /// Every executed encrypt (`true`) or decrypt as `(step, group, block, encrypt)`.
    pub fn crypt_log(&self) -> &[(u64, u32, u32, bool)] {
        &self.crypt_log
    }

    /// Every executed demote (`true`) or promote as `(step, group, block, host, demote)`.
    pub fn transfer_log(&self) -> &[(u64, u32, u32, u32, bool)] {
        &self.transfer_log
    }

    /// Every executed host zero as `(step, group, host)`.
    pub fn host_zero_log(&self) -> &[(u64, u32, u32)] {
        &self.host_zero_log
    }

    pub fn block_is_zero(&self, group: u32, block: u32) -> bool {
        if group == self.state_pool() {
            return self.states.is_zero(block);
        }
        self.pools[group as usize].block_is_zero(block)
    }

    pub fn block_is_encrypted(&self, group: u32, block: u32) -> bool {
        if group == self.state_pool() {
            return self.states.is_encrypted(block);
        }
        self.pools[group as usize].is_encrypted(block)
    }

    /// A block's (or state buffer's) image as words, in its cipher's order.
    pub fn block_image(&self, group: u32, block: u32) -> Vec<u32> {
        if group == self.state_pool() {
            return self.states.image(block);
        }
        self.pools[group as usize].image(block)
    }

    fn host_states(&self) -> &StatePool {
        self.host_states.as_ref().expect("a host tier")
    }

    pub fn host_block_image(&self, group: u32, host: u32) -> Vec<u32> {
        if group == self.state_pool() {
            return self.host_states().image(host);
        }
        self.host_pools[group as usize].image(host)
    }

    pub fn host_block_is_zero(&self, group: u32, host: u32) -> bool {
        if group == self.state_pool() {
            return self.host_states().is_zero(host);
        }
        self.host_pools[group as usize].block_is_zero(host)
    }

    pub fn host_block_is_encrypted(&self, group: u32, host: u32) -> bool {
        if group == self.state_pool() {
            return self.host_states().is_encrypted(host);
        }
        self.host_pools[group as usize].is_encrypted(host)
    }

    /// Whether a slot holds no per-sequence state: no block mapped, no base, no tail and
    /// no drafter state.
    pub fn slot_state_is_zero(&self, slot: Slot) -> bool {
        let s = slot as usize;
        self.tables[s].iter().flatten().all(|&b| b == NULL_BLOCK)
            && self.tails[s].is_zero()
            && self.hiddens[s].is_zero()
    }

    /// Rows recorded since the last call (with [`QwenCpuExecutorConfig::record`]).
    pub fn take_records(&self) -> Vec<QwenRowRecord> {
        std::mem::take(&mut *self.records.lock().expect("records lock"))
    }

    /// States stored since the last call (with [`QwenCpuExecutorConfig::record`]).
    pub fn take_stored(&self) -> Vec<StoredState> {
        std::mem::take(&mut *self.stored.lock().expect("stored lock"))
    }

    /// Bytes one KV block of `group` takes, and one state buffer (the node's sizing).
    pub fn block_bytes(&self, group: u32) -> usize {
        if group == self.state_pool() {
            return self.states.buffer_bytes();
        }
        self.pools[group as usize].block_bytes()
    }

    fn set_table(&mut self, slot: Slot, group: u32, index: u32, block: u32) {
        assert!(
            (group as usize) < self.pools.len(),
            "a table update names group {group}, which pages no KV"
        );
        self.tables[slot as usize][group as usize][index as usize] = block;
    }

    fn reset_slot(&mut self, slot: Slot) {
        let s = slot as usize;
        for g in &mut self.tables[s] {
            g.fill(NULL_BLOCK);
        }
        self.tails[s].reset();
        self.hiddens[s].reset();
    }

    fn block_of(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        let b = self.tables[slot as usize][group][(pos / self.cfg.block_size) as usize];
        assert_ne!(
            b, NULL_BLOCK,
            "slot {slot} group {group} position {pos}: unmapped block"
        );
        b
    }

    /// A block this slot may write: mapped by no other slot.
    fn writable_block(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        let b = self.block_of(slot, group, pos);
        let idx = (pos / self.cfg.block_size) as usize;
        let shared = self
            .tables
            .iter()
            .enumerate()
            .any(|(s, t)| s != slot as usize && t[group][idx] == b);
        assert!(
            !shared,
            "slot {slot} group {group} position {pos}: write to a shared block"
        );
        b
    }

    /// Tokens at positions `0..n`, read back from the target group's row tags.
    fn prefix_tokens(&self, slot: Slot, n: u32) -> Vec<u32> {
        let bs = self.cfg.block_size;
        (0..n)
            .map(|pos| {
                let b = self.block_of(slot, TARGET, pos);
                let tag = self.pools[TARGET].tag(b, 0, (pos % bs) as usize);
                assert_eq!(tag.pos_plus_one, pos + 1, "slot {slot}: tag at {pos}");
                tag.token
            })
            .collect()
    }

    /// Layer `layer`'s KV of group `group` at positions `from..to` as a contiguous cache,
    /// checking each row's tag.
    fn gather(&self, slot: Slot, group: usize, layer: usize, from: u32, to: u32) -> KvCache {
        let bs = self.cfg.block_size;
        let mut cache = KvCache::default();
        for pos in from..to {
            let b = self.block_of(slot, group, pos);
            let (k, v, tag) = self.pools[group].read(b, layer, (pos % bs) as usize);
            assert_eq!(
                tag.pos_plus_one,
                pos + 1,
                "slot {slot} group {group} layer {layer} position {pos}: read of KV that was \
                 never written, zeroed, encrypted or foreign"
            );
            cache.keys.extend_from_slice(k);
            cache.values.extend_from_slice(v);
        }
        cache
    }

    /// Stores `cache`'s entries `skip..` at positions `from..` of layer `layer`, tagged
    /// with `tokens`.
    #[allow(clippy::too_many_arguments)]
    fn scatter(
        &mut self,
        slot: Slot,
        group: usize,
        layer: usize,
        from: u32,
        cache: &KvCache,
        skip: usize,
        tokens: &[u32],
    ) {
        let bs = self.cfg.block_size;
        let w = self.pools[group].layout.k_width();
        for (i, &token) in tokens.iter().enumerate() {
            let pos = from + i as u32;
            let e = skip + i;
            let b = self.writable_block(slot, group, pos);
            self.pools[group].write(
                b,
                layer,
                (pos % bs) as usize,
                Tag {
                    pos_plus_one: pos + 1,
                    token,
                },
                &cache.keys[e * w..(e + 1) * w],
                &cache.values[e * w..(e + 1) * w],
            );
        }
    }

    /// The DeltaNet layers' states at `len` for `slot`: its base, folded over its tail.
    fn fold(&self, slot: Slot, len: u32) -> Vec<DeltaNetState> {
        let tail = &self.tails[slot as usize];
        let (base, base_len) = tail.base.expect("a slot with a base");
        assert!(
            len >= base_len && (len - base_len) as usize <= tail.len(),
            "slot {slot}: a fold to {len} over positions the slot never kept (base {base_len}, \
             {} kept)",
            tail.len()
        );
        let mut states = self.load_base(base, base_len);
        for pos in base_len..len {
            self.advance(&mut states, &tail.entry(pos));
        }
        states
    }

    fn load_base(&self, base: u32, base_len: u32) -> Vec<DeltaNetState> {
        if base == NULL_STATE {
            assert_eq!(
                base_len, 0,
                "the zero state named as the state at {base_len}"
            );
            let spec = &self.model.config().deltanet;
            return self
                .deltanet
                .iter()
                .map(|_| DeltaNetState::new(spec))
                .collect();
        }
        self.states.read(base, base_len)
    }

    /// One position through every DeltaNet layer's recurrence, from its kept inputs: the
    /// model's own per-token core, whose output is not needed.
    fn advance(&self, states: &mut [DeltaNetState], entry: &TailEntry) {
        let c = self.model.config();
        let zeros = vec![0.0f32; c.deltanet.value_dim()];
        for (di, (s, (mixed, a, b))) in states.iter_mut().zip(entry).enumerate() {
            let w = self.deltanet_weights(self.deltanet[di]);
            let core = CoreWeights {
                conv: &w.conv,
                a_log: &w.a_log,
                dt_bias: &w.dt_bias,
                norm: &w.norm,
            };
            token_core(&c.deltanet, s, mixed, &zeros, a, b, &core, c.rms_norm_eps);
        }
    }

    fn layer(&self, l: usize) -> &LayerWeights {
        &self.model.weights.layers[l]
    }

    fn deltanet_weights(&self, l: usize) -> &eidola_engine_model::qwen::weights::DeltaNetWeights {
        match &self.layer(l).mixer {
            MixerWeights::DeltaNet(w) => w,
            MixerWeights::Attention(_) => panic!("layer {l} is an attention layer"),
        }
    }

    /// Stores the states at `len` into `buffer`.
    fn store(&mut self, slot: Slot, buffer: u32, len: u32, states: &[DeltaNetState]) {
        self.states.write(buffer, len, states);
        if self.cfg.record {
            let tokens = self.prefix_tokens(slot, len);
            self.stored.lock().expect("stored lock").push(StoredState {
                slot,
                len,
                tokens,
                states: states.to_vec(),
            });
        }
    }

    /// The target over `toks` at positions `start..`: every layer, from `states` (the
    /// DeltaNet layers' states at `start`), writing the attention layers' KV into the
    /// slot's pages. Returns the normed final rows and, per position, its recurrence
    /// inputs; `snaps` gets the states at each boundary in `writes` the rows pass.
    fn target(
        &mut self,
        slot: Slot,
        start: u32,
        toks: &[u32],
        states: &mut [DeltaNetState],
        writes: &[u32],
        snaps: &mut BTreeMap<u32, Vec<DeltaNetState>>,
    ) -> Result<(Matrix, Vec<TailEntry>), ExecutorError> {
        let model = self.model.clone();
        let c = model.config();
        let eps = c.rms_norm_eps;
        let n = toks.len();
        let mut h = model.embed(toks).map_err(err)?;
        let mut inputs: Vec<TailEntry> = vec![Vec::new(); n];
        let mut di = 0;
        let mut ai = 0;
        for (l, spec) in c.layers.iter().enumerate() {
            let lw = &model.weights.layers[l];
            match (spec.kind, &lw.mixer) {
                (LayerKind::DeltaNet, MixerWeights::DeltaNet(dw)) => {
                    // The model's DeltaNet layer, its projections kept for the tail.
                    let x = rms_norm_zc_rows(&h, &lw.input_norm, eps);
                    let mixed = lin8_rows(&x, &dw.in_proj_qkv);
                    let z = lin8_rows(&x, &dw.in_proj_z);
                    let b = linear_bf16_dense(&x, &dw.in_proj_b);
                    let a = linear_bf16_dense(&x, &dw.in_proj_a);
                    let core = CoreWeights {
                        conv: &dw.conv,
                        a_log: &dw.a_log,
                        dt_bias: &dw.dt_bias,
                        norm: &dw.norm,
                    };
                    let mut o = Matrix::zeros(n, c.deltanet.value_dim());
                    for (r, kept) in inputs.iter_mut().enumerate() {
                        let out = token_core(
                            &c.deltanet,
                            &mut states[di],
                            mixed.row(r),
                            z.row(r),
                            a.row(r),
                            b.row(r),
                            &core,
                            eps,
                        );
                        o.row_mut(r).copy_from_slice(&out);
                        kept.push((mixed.row(r).to_vec(), a.row(r).to_vec(), b.row(r).to_vec()));
                        let len = start + r as u32 + 1;
                        if writes.contains(&len) {
                            snaps
                                .entry(len)
                                .or_insert_with(|| Vec::with_capacity(states.len()))
                                .push(states[di].clone());
                        }
                    }
                    let mix = lin8_rows(&o, &dw.out_proj);
                    let mut h2 = h.clone();
                    residual_add(&mut h2, &mix);
                    let x2 = rms_norm_zc_rows(&h2, &lw.post_norm, eps);
                    let m = model.mlp(&lw.mlp, &x2);
                    residual_add(&mut h2, &m);
                    h = h2;
                    di += 1;
                }
                (LayerKind::Attention, MixerWeights::Attention(_)) => {
                    let cache = self.gather(slot, TARGET, ai, 0, start);
                    let mut st = LayerState::Attention(cache);
                    h = model
                        .layer_forward(l, &mut st, &h, start as usize)
                        .map_err(err)?;
                    let LayerState::Attention(cache) = st else {
                        unreachable!("an attention layer's state")
                    };
                    self.scatter(slot, TARGET, ai, start, &cache, start as usize, toks);
                    ai += 1;
                }
                _ => return Err(ExecutorError(format!("layer {l}: weights of another kind"))),
            }
        }
        Ok((model.final_norm(&h), inputs))
    }

    /// The drafter's rows at positions `from..from + toks.len()` (`from >= 1`): row `s`
    /// consumes the token at `s` and `hidden` row `s - from` (the target's normed state at
    /// `s - 1`), its KV written at `s`. Returns its logits at `logits` (offsets) and its
    /// normed output.
    fn drafter_rows(
        &mut self,
        slot: Slot,
        from: u32,
        toks: &[u32],
        hidden: &Matrix,
        logits: &[usize],
    ) -> Result<(Matrix, Matrix), ExecutorError> {
        assert!(from >= 1, "no drafter row at position 0");
        let model = self.model.clone();
        // The drafter's KV at position `s` is the model's MTP cache entry `s - 1`.
        let mut cache = self.gather(slot, DRAFTER, 0, 1, from);
        let out = model
            .mtp_forward(
                0,
                &mut cache,
                (from - 1) as usize,
                toks,
                hidden,
                &LogitRows::At(logits.to_vec()),
            )
            .map_err(err)?;
        self.scatter(slot, DRAFTER, 0, from, &cache, (from - 1) as usize, toks);
        Ok((out.logits, out.normed))
    }

    /// The target's normed state at `pos` for `slot`: from what its last row left, or
    /// from the boundary tap of the drafter block ending there (after a resume).
    fn hidden_before(&self, slot: Slot, pos: u32) -> Vec<f32> {
        if let Some(h) = self.hiddens[slot as usize].get(pos) {
            return h;
        }
        let b = self.block_of(slot, DRAFTER, pos);
        self.pools[DRAFTER]
            .read_tap(b, pos)
            .unwrap_or_else(|| panic!("slot {slot}: no drafter state at {pos}"))
            .to_vec()
    }

    fn run_row(&mut self, mut row: Row) -> Result<Row, ExecutorError> {
        let slot = row.slot;
        let bs = self.cfg.block_size;
        let StateRow { base, base_len } = row.base;
        // The slot's base is the one its last row left, or a fresh slot starts at it.
        match self.tails[slot as usize].base {
            Some(b) => assert_eq!(
                b,
                (base, base_len),
                "slot {slot}: the row names base {base}@{base_len}, the slot's is {}@{}",
                b.0,
                b.1
            ),
            None => {
                assert_eq!(
                    row.c, base_len,
                    "slot {slot}: a fresh slot starts at its base"
                );
                self.tails[slot as usize].set((base, base_len), &[]);
            }
        }
        assert!(
            base_len.is_multiple_of(bs) && base_len <= row.c,
            "slot {slot}: a base at {base_len} is no boundary at most {}",
            row.c
        );
        let e_end = row.p + 1;
        let write_lens: Vec<u32> = row.writes.iter().map(|w| w.len).collect();
        for (i, w) in row.writes.iter().enumerate() {
            assert!(
                w.len > base_len && w.len <= e_end && w.len.is_multiple_of(bs),
                "slot {slot}: a state write at {} outside ({base_len}, {e_end}]",
                w.len
            );
            assert!(
                i == 0 || row.writes[i - 1].len < w.len,
                "state writes out of order"
            );
            assert!(
                w.buffer != NULL_STATE && (w.buffer as usize) < self.cfg.state_buffers as usize,
                "a state write into buffer {}",
                w.buffer
            );
            if w.buffer == base {
                assert_eq!(
                    i + 1,
                    row.writes.len(),
                    "slot {slot}: its base overwritten early"
                );
            } else {
                assert!(
                    self.states.is_zero(w.buffer),
                    "state buffer {}: written while it holds a state",
                    w.buffer
                );
            }
        }

        // 1. The states at `c`: the base folded over the tail, storing the boundaries the
        //    fold passes.
        let mut snaps: BTreeMap<u32, Vec<DeltaNetState>> = BTreeMap::new();
        let mut states = self.load_base(base, base_len);
        let kept: Vec<TailEntry> = (base_len..row.c)
            .map(|pos| self.tails[slot as usize].entry(pos))
            .collect();
        for (i, entry) in kept.iter().enumerate() {
            self.advance(&mut states, entry);
            let len = base_len + i as u32 + 1;
            if write_lens.contains(&len) {
                snaps.insert(len, states.clone());
            }
        }

        // 2. The target over the host tokens.
        let toks = row.toks.clone();
        let (normed_a, inputs_a) =
            self.target(slot, row.c, &toks, &mut states, &write_lens, &mut snaps)?;
        for w in &row.writes {
            let s = snaps
                .remove(&w.len)
                .unwrap_or_else(|| panic!("slot {slot}: no state at {}", w.len));
            self.store(slot, w.buffer, w.len, &s);
        }
        let host_logit_rows: Vec<usize> = if self.cfg.record {
            (0..toks.len()).collect()
        } else if row.sample {
            vec![toks.len() - 1]
        } else {
            Vec::new()
        };
        let logits_a = self.model.head(&normed_a.select_rows(&host_logit_rows));
        for (i, &r) in host_logit_rows.iter().enumerate() {
            row.target_logits
                .insert(row.c + r as u32, logits_a.row(i).to_vec());
        }
        let (c0, p0) = (row.c, row.p);
        let normed_row = move |pos: u32, a: &Matrix, b: &Matrix| -> Vec<f32> {
            if pos <= p0 {
                a.row((pos - c0) as usize).to_vec()
            } else {
                b.row((pos - e_end) as usize).to_vec()
            }
        };

        // 3. The drafter's rows over the host positions.
        let drafting = self.pools.len() > DRAFTER;
        let mut draft_normed = None;
        if drafting {
            let from = row.c.max(1);
            if from <= row.p {
                let mut hidden = Matrix::zeros((e_end - from) as usize, normed_a.cols);
                for (i, s) in (from..e_end).enumerate() {
                    let h = if s == row.c {
                        self.hidden_before(slot, s - 1)
                    } else {
                        normed_a.row((s - 1 - row.c) as usize).to_vec()
                    };
                    hidden.row_mut(i).copy_from_slice(&h);
                }
                let n = (e_end - from) as usize;
                let wanted: Vec<usize> = if self.cfg.record {
                    (0..n).collect()
                } else if row.k > 0 {
                    vec![n - 1]
                } else {
                    Vec::new()
                };
                let dtoks = toks[(from - row.c) as usize..].to_vec();
                let (dl, dn) = self.drafter_rows(slot, from, &dtoks, &hidden, &wanted)?;
                for (i, &r) in wanted.iter().enumerate() {
                    row.drafter_logits
                        .push((from + r as u32, dl.row(i).to_vec()));
                }
                if row.k > 0 {
                    let last = wanted
                        .iter()
                        .position(|&r| r == n - 1)
                        .expect("the last row");
                    row.chain_logits.push(dl.row(last).to_vec());
                    draft_normed = Some(dn.select_rows(&[n - 1]));
                }
            }
        }

        // 4. The draft chain: the row at `p` drafts the first token; the layer then
        //    recurses on its own normed output, in a cache of its own.
        if row.k > 0 {
            let model = self.model.clone();
            let mut cache = self.gather(slot, DRAFTER, 0, 1, e_end);
            let mut prev = draft_normed.expect("the row at p");
            for i in 0..row.k as usize {
                let logits = row.chain_logits[i].clone();
                self.draw(&mut row, &logits);
                if i + 1 == row.k as usize {
                    break;
                }
                let d = row.drafts[i];
                let out = model
                    .mtp_forward(
                        0,
                        &mut cache,
                        row.p as usize + i,
                        &[d],
                        &prev,
                        &LogitRows::Last,
                    )
                    .map_err(err)?;
                row.chain_logits.push(out.logits.row(0).to_vec());
                prev = out.normed;
            }
        }

        // 5. The target over the drafts, from the states the host tokens left; nothing it
        //    computes is stored but its KV in the reserved positions.
        let drafts = row.drafts.clone();
        let (normed_b, inputs_b) = if drafts.is_empty() {
            (Matrix::zeros(0, normed_a.cols), Vec::new())
        } else {
            let mut none = BTreeMap::new();
            self.target(slot, e_end, &drafts, &mut states, &[], &mut none)?
        };
        if !drafts.is_empty() {
            let rows: Vec<usize> = (0..drafts.len()).collect();
            let logits_b = self.model.head(&normed_b.select_rows(&rows));
            for i in rows {
                row.target_logits
                    .insert(e_end + i as u32, logits_b.row(i).to_vec());
            }
        }

        // 6. Sampling and chain acceptance.
        if row.sample {
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
        let accepted = row.produced.len().saturating_sub(1) as u32;

        // 7. The drafter's rows at the accepted drafts, rewritten from the target's states:
        //    the chain wrote them from its own outputs, and a sealed block must hold the
        //    canonical rows.
        if drafting && accepted > 0 {
            let mut hidden = Matrix::zeros(accepted as usize, normed_a.cols);
            for i in 0..accepted {
                let h = normed_row(row.p + i, &normed_a, &normed_b);
                hidden.row_mut(i as usize).copy_from_slice(&h);
            }
            let wanted: Vec<usize> = if self.cfg.record {
                (0..accepted as usize).collect()
            } else {
                Vec::new()
            };
            let dtoks = drafts[..accepted as usize].to_vec();
            let (dl, _) = self.drafter_rows(slot, e_end, &dtoks, &hidden, &wanted)?;
            for (i, &r) in wanted.iter().enumerate() {
                row.drafter_logits
                    .push((e_end + r as u32, dl.row(i).to_vec()));
            }
        }

        // 8. Drafter state and boundary taps: the target's normed states at the positions
        //    the next row may continue from, and at every block's last position computed.
        if drafting {
            let last = row.p + accepted;
            for q in row.c..=last {
                if (q + 1).is_multiple_of(bs) {
                    let b = self.writable_block(slot, DRAFTER, q);
                    let h = normed_row(q, &normed_a, &normed_b);
                    self.pools[DRAFTER].write_tap(b, q, &h);
                }
            }
            let keep: Vec<(u32, Vec<f32>)> = (row.p..=row.p + row.k)
                .map(|pos| (pos, normed_row(pos, &normed_a, &normed_b)))
                .collect();
            self.hiddens[slot as usize].set(&keep);
        }

        // 9. The tail: every position past the new base, drafts included.
        let (new_base, new_len) = row
            .writes
            .last()
            .map_or((base, base_len), |w| (w.buffer, w.len));
        let mut all: Vec<TailEntry> = kept;
        all.extend(inputs_a);
        all.extend(inputs_b);
        let tail: Vec<TailEntry> = all.split_off((new_len - base_len) as usize);
        self.tails[slot as usize].set((new_base, new_len), &tail);
        Ok(row)
    }

    /// Draws the next draft from the drafter's `logits`.
    fn draw(&self, row: &mut Row, logits: &[f32]) {
        let logits = Logits::new(logits, self.spec.sampleable_vocab_size);
        let pos = row.p as u64 + 1 + row.drafts.len() as u64;
        let (d, q) = if row.params.is_greedy() {
            let d = sampling::argmax(logits);
            let mut q = vec![0.0; logits.len()];
            q[d as usize] = 1.0;
            (d, q)
        } else {
            let q = sampling::processed_probs(logits, &row.params);
            let d =
                sampling::sample_from(&q, sampling::uniform(row.params.seed(), pos, Stream::Draft));
            (d, q)
        };
        row.drafts.push(d);
        row.q_rows.push(q);
    }

    fn maintain(&mut self, step: &StepInput) {
        let state = self.state_pool();
        for m in &step.maintenance {
            match *m {
                Maintenance::SaveState { slot, len, buffer } => {
                    let tail = &self.tails[slot as usize];
                    let (base, _) = tail
                        .base
                        .expect("a state saved from a slot that holds none");
                    assert!(
                        len.is_multiple_of(self.cfg.block_size),
                        "a state saved off a boundary"
                    );
                    assert!(
                        buffer == base || self.states.is_zero(buffer),
                        "state buffer {buffer}: saved into while it holds a state"
                    );
                    let states = self.fold(slot, len);
                    self.store(slot, buffer, len, &states);
                }
                Maintenance::ResetSlot { slot } => self.reset_slot(slot),
                Maintenance::Copy { group, src, dst } => {
                    assert_ne!(group, state, "a copy names the state pool");
                    self.pools[group as usize].copy(src, dst);
                }
                Maintenance::Zero { group, block } => {
                    if group == state {
                        self.states.zero(block);
                    } else {
                        self.pools[group as usize].zero(block);
                    }
                    self.zero_log.push((self.steps, group, block));
                }
                Maintenance::Encrypt {
                    group,
                    block,
                    key,
                    nonce,
                }
                | Maintenance::Decrypt {
                    group,
                    block,
                    key,
                    nonce,
                } => {
                    let encrypt = matches!(m, Maintenance::Encrypt { .. });
                    let key = step
                        .kv_keys
                        .get(key as usize)
                        .unwrap_or_else(|| panic!("maintenance names key {key}, not in the step"));
                    if group == state {
                        self.states.crypt(block, key, &nonce, encrypt);
                    } else {
                        self.pools[group as usize].crypt(block, key, &nonce, encrypt);
                    }
                    self.crypt_log.push((self.steps, group, block, encrypt));
                }
                Maintenance::Demote { group, block, host } => {
                    if group == state {
                        assert!(
                            self.states.is_encrypted(block),
                            "state buffer {block}: demote of a plaintext buffer"
                        );
                        let to = self.host_states.as_mut().expect("a host tier");
                        StatePool::transfer(&self.states, block, to, host);
                    } else {
                        let g = group as usize;
                        assert!(
                            self.pools[g].is_encrypted(block),
                            "group {group} block {block}: demote of a plaintext block"
                        );
                        Pool::transfer(&self.pools[g], block, &mut self.host_pools[g], host);
                    }
                    self.transfer_log
                        .push((self.steps, group, block, host, true));
                }
                Maintenance::Promote { group, host, block } => {
                    if group == state {
                        let from = self.host_states.as_ref().expect("a host tier");
                        assert!(
                            from.is_encrypted(host),
                            "state host buffer {host}: promote of a buffer holding no ciphertext"
                        );
                        assert!(
                            self.states.is_zero(block),
                            "state buffer {block}: promote into a buffer that is not zero"
                        );
                        StatePool::transfer(from, host, &mut self.states, block);
                    } else {
                        let g = group as usize;
                        assert!(
                            self.host_pools[g].is_encrypted(host),
                            "group {group} host block {host}: promote of a block holding no ciphertext"
                        );
                        assert!(
                            self.pools[g].block_is_zero(block)
                                && !self.pools[g].is_encrypted(block),
                            "group {group} block {block}: promote into a block that is not zero"
                        );
                        Pool::transfer(&self.host_pools[g], host, &mut self.pools[g], block);
                    }
                    self.transfer_log
                        .push((self.steps, group, block, host, false));
                }
                Maintenance::ZeroHost { group, host } => {
                    if group == state {
                        self.host_states.as_mut().expect("a host tier").zero(host);
                    } else {
                        self.host_pools[group as usize].zero(host);
                    }
                    self.host_zero_log.push((self.steps, group, host));
                }
            }
        }
    }
}

fn err(e: eidola_engine_model::Error) -> ExecutorError {
    ExecutorError(e.to_string())
}

fn refuse_lock(e: std::io::Error) -> ! {
    panic!(
        "cannot lock the host tier into memory ({e}); raise the memory-lock limit \
         (RLIMIT_MEMLOCK) or give the executor no host tier"
    )
}

impl Executor for QwenCpuExecutor {
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
        self.maintain(step);
        for u in &step.table_updates {
            self.set_table(u.slot, u.group, u.index, u.block);
        }
        assert_eq!(
            step.aux.state_rows.len(),
            step.seqs.len(),
            "every row names its state base"
        );
        let mut by_row: HashMap<u32, Vec<StateWrite>> = HashMap::new();
        let mut written = std::collections::HashSet::new();
        for w in &step.aux.state_writes {
            assert!(
                written.insert(w.buffer),
                "state buffer {} written twice",
                w.buffer
            );
            by_row.entry(w.row).or_default().push(*w);
        }
        for (i, r) in step.aux.state_rows.iter().enumerate() {
            if r.base != NULL_STATE {
                let foreign = step
                    .aux
                    .state_writes
                    .iter()
                    .any(|w| w.buffer == r.base && w.row as usize != i);
                assert!(
                    !foreign,
                    "state buffer {}: written while another row reads it",
                    r.base
                );
            }
        }

        let mut rows: Vec<Row> = Vec::with_capacity(step.seqs.len());
        for (i, e) in step.seqs.iter().enumerate() {
            rows.push(self.row_of(step, i, e, by_row.remove(&(i as u32)).unwrap_or_default()));
        }
        let mut done = Vec::with_capacity(rows.len());
        for row in rows {
            done.push(self.run_row(row)?);
        }

        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; done.len() * stride as usize],
            stride,
            num_tokens: vec![0; done.len()],
            logits: step.return_logits.then(Vec::new),
        };
        for (i, row) in done.iter().enumerate() {
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
            for row in done {
                let mut tokens = self.prefix_tokens(row.slot, row.c);
                tokens.extend(&row.toks);
                tokens.extend(&row.drafts);
                self.records
                    .lock()
                    .expect("records lock")
                    .push(QwenRowRecord {
                        slot: row.slot,
                        context_len: row.c,
                        num_tokens: row.p + 1 - row.c,
                        tokens,
                        target_logits: row.target_logits.into_iter().collect(),
                        drafter_logits: row.drafter_logits,
                        chain_logits: row.chain_logits,
                        drafts: row.drafts,
                        produced: row.produced,
                        sampling: row.params,
                    });
            }
        }
        Ok(out)
    }
}

impl QwenCpuExecutor {
    fn row_of(&self, step: &StepInput, i: usize, e: &SeqEntry, writes: Vec<StateWrite>) -> Row {
        assert!(e.num_tokens >= 1, "a row needs at least one host token");
        let start = e.token_start as usize;
        let end = start + e.num_tokens as usize;
        for (j, &pos) in step.positions[start..end].iter().enumerate() {
            assert_eq!(
                pos,
                e.context_len + j as u32,
                "positions disagree with context_len"
            );
        }
        let p = e.context_len + e.num_tokens - 1;
        let k = if e.sample { e.num_drafts } else { 0 };
        assert!(
            k <= self.cfg.draft_tokens,
            "{k} drafts requested; the executor drafts {}",
            self.cfg.draft_tokens
        );
        assert!(
            p > 0 || k == 0,
            "slot {}: {k} drafts requested at position 0",
            e.slot
        );
        Row {
            slot: e.slot,
            c: e.context_len,
            p,
            k,
            sample: e.sample,
            params: e.sampling,
            toks: step.token_ids[start..end].to_vec(),
            base: step.aux.state_rows[i],
            writes,
            drafts: Vec::new(),
            q_rows: Vec::new(),
            produced: Vec::new(),
            target_logits: BTreeMap::new(),
            drafter_logits: Vec::new(),
            chain_logits: Vec::new(),
        }
    }
}

/// The floats one tail position takes over every DeltaNet layer of `model` (the node's
/// sizing).
pub fn tail_floats_per_position(model: &QwenModel) -> usize {
    let c = model.config();
    c.layers_of(LayerKind::DeltaNet).count() * tail_floats_per_layer(&c.deltanet)
}
