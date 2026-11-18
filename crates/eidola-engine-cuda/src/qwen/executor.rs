//! [`QwenCudaExecutor`]: the serving core's [`Executor`] for Qwen3.8 on a [`QwenDevice`].
//!
//! It reproduces `eidola-engine-cpu`'s `QwenCpuExecutor` (the contract every device
//! executor is held to), step for step: the same maintenance, the same rows, the same
//! state images byte for byte, and in the exact profile the same logits bit for bit. What
//! differs is only where the numbers are computed (the kernel families of
//! `csrc/qwen_abi.cuh`, batched over a step's rows) and how the recurrent state between
//! steps is kept ([`StateStrategy`]).
//!
//! A step is one launch sequence over fixed scratch ([`program`]), reading every per-step
//! value from one table ([`super::plan`]):
//!
//! 1. Maintenance in order, then table updates; the rows checked and planned.
//! 2. With drafting, the drafter at each drafting row's one host position `p` (the token
//!    at `p` and the target's normed state at `p - 1`, which the slot kept or a tap holds:
//!    both from earlier steps), then the draft chain, depth by depth over every drafting
//!    row, each depth's drafts drawn on the device.
//! 3. **The target, once**, over every row's host tokens followed by its drafts, all rows
//!    in one forward. A DeltaNet layer's recurrence starts each row from its source (the
//!    slot's live state, or its base buffer), folds the kept positions between the source
//!    and the row's `context_len` without output, runs the host tokens and the drafts with
//!    output, stores every named state write as the recurrence passes it and the state
//!    after the host tokens into the slot's live image; the new positions' inputs join the
//!    tail, and the drafts' KV goes into the positions reserved for them.
//! 4. The drafter's KV over the host positions of rows that do not draft.
//! 5. Plain rows drawn and chain acceptance on the device (the shared sampler, equal to
//!    `eidola_engine::sampling` bit for bit given equal logits).
//! 6. The drafter's canonical KV at every draft's position (a rejected draft's is
//!    rewritten when its position is computed again, before anything reads it), the
//!    boundary taps and the kept normed states.
//!
//! The step's one read-back is the status word and the tokens. A uniform decode step runs
//! a rung's padded launch sequence: replayed from its captured graph, or directly; any
//! other step runs the sequence laid out for itself (the eager path).

#[path = "program.rs"]
mod program;

use std::any::Any;
use std::cell::Cell;
use std::collections::HashMap;

use eidola_engine::executor::{
    Executor, ExecutorError, Maintenance, SeqEntry, Slot, StateRow, StateWrite, StepInput,
    StepOutput,
};
use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::KvKey;
use eidola_engine::spec::{
    AttentionKind, Bucket, Coverage, KvGroupSpec, KvRole, ModelSpec, NULL_BLOCK, NULL_STATE,
    StateGroupSpec, StateSpec,
};
use eidola_engine_model::qwen::QwenModel;
use eidola_engine_model::qwen::config::{LayerKind, QwenConfig};

use super::abi::*;
use super::device::{CipherRange, QwenDevice, QwenProfile};
use super::plan::{
    self, AttnRows, DrafterPass, Fa2Geometry, Geo, Item, Layout, SAMPLE_WORDS, Shape, StepPlan,
    TargetPass, buf,
};
use super::weights::DevWeights;
use crate::mimo::attention::{PassCaps, Reduction};
pub use crate::mimo::graph::{CudaGraphs, DecodePath};
use crate::sampler::SampleRow;
use crate::{CudaError, Result, narrow};
use program::{Caps, Scratch};

/// KV group of the target's attention layers.
const TARGET: usize = 0;
/// KV group of the drafter's layer, when drafting.
const DRAFTER: usize = 1;

/// How the executor reaches a row's DeltaNet states at its `context_len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StateStrategy {
    /// A live image per slot, at the end of its last row's host tokens: a row folds only
    /// the drafts accepted since. The serving form.
    #[default]
    Live,
    /// Every row loads its base buffer and folds the slot's whole tail, as the reference
    /// executor does. For measurement: the same bits.
    Lagging,
}

impl StateStrategy {
    /// `live` or `lagging`, exactly.
    pub fn parse(s: &str) -> Option<StateStrategy> {
        match s {
            "live" => Some(StateStrategy::Live),
            "lagging" => Some(StateStrategy::Lagging),
            _ => None,
        }
    }
}

/// Executor configuration: memory geometry, drafting and the state strategy, as the
/// reference executor's (`QwenCpuExecutorConfig`) plus [`StateStrategy`].
#[derive(Clone, Debug, PartialEq)]
pub struct QwenCudaExecutorConfig {
    pub block_size: u32,
    /// Physical blocks per KV group, including the null block 0.
    pub num_blocks: u32,
    /// Host-tier blocks per KV group, including the null block 0; 0 for none.
    pub host_blocks: u32,
    /// State buffers, including the reserved zero state 0.
    pub state_buffers: u32,
    /// Host-tier state buffers, including a reserved buffer 0; 0 for none.
    pub host_state_buffers: u32,
    pub num_state_slots: u32,
    pub max_model_len: u32,
    pub buckets: Vec<Bucket>,
    pub draft_tokens: u32,
    pub sampleable_vocab_size: u32,
    pub strategy: StateStrategy,
    /// Whether uniform decode steps replay graphs captured at construction.
    pub graphs: CudaGraphs,
}

/// A device allocation and what keeps it.
struct Alloc {
    addr: u64,
    _keep: Box<dyn Any>,
}

impl Alloc {
    fn new<D: QwenDevice>(dev: &D, bytes: usize) -> Result<Alloc> {
        let (addr, keep) = dev.alloc(bytes.max(8))?;
        Ok(Alloc { addr, _keep: keep })
    }
}

/// Host-tier memory.
struct HostAlloc {
    ptr: *mut u8,
    _keep: Box<dyn Any>,
}

/// The model's dimensions, as the kernels take them.
#[derive(Clone, Debug)]
struct Dims {
    hidden: u32,
    inter: u32,
    vocab: u32,
    eps: f32,
    kinds: Vec<LayerKind>,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary: u32,
    scale: f32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
}

impl Dims {
    fn of(c: &QwenConfig) -> Result<Dims> {
        let a = &c.attention;
        let d = &c.deltanet;
        Ok(Dims {
            hidden: u32of(c.hidden_size, "hidden")?,
            inter: u32of(c.intermediate_size, "intermediate")?,
            vocab: u32of(c.vocab_size, "vocabulary")?,
            eps: c.rms_norm_eps,
            kinds: c.layers.iter().map(|l| l.kind).collect(),
            q_heads: u32of(a.num_q_heads, "query heads")?,
            kv_heads: u32of(a.num_kv_heads, "kv heads")?,
            head_dim: u32of(a.head_dim, "head dim")?,
            rotary: u32of(a.rotary_dim, "rotary")?,
            scale: a.softmax_scale,
            k_heads: u32of(d.num_k_heads, "key heads")?,
            v_heads: u32of(d.num_v_heads, "value heads")?,
            k_dim: u32of(d.k_head_dim, "key dim")?,
            v_dim: u32of(d.v_head_dim, "value dim")?,
        })
    }

    /// Floats of a position's entry in the blocked DeltaNet's tail cache, a layer.
    fn chunk_entry(&self) -> usize {
        chunk_entry_floats(self.k_heads, self.v_heads, self.k_dim, self.v_dim)
    }

    /// The widest K of an FP8 linear.
    fn k_max(&self) -> usize {
        [
            self.hidden,
            self.inter,
            self.q_heads * self.head_dim,
            self.value_dim(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0) as usize
    }
    fn key_dim(&self) -> u32 {
        self.k_heads * self.k_dim
    }
    fn value_dim(&self) -> u32 {
        self.v_heads * self.v_dim
    }
    fn conv_dim(&self) -> u32 {
        2 * self.key_dim() + self.value_dim()
    }
    fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
    /// f32 values of one DeltaNet layer in a state image: the conv history, then `S`.
    fn layer_floats(&self) -> usize {
        3 * self.conv_dim() as usize + (self.v_heads * self.k_dim * self.v_dim) as usize
    }
}

/// A slot's state between steps.
#[derive(Clone, Debug, Default)]
struct SlotRec {
    /// `(buffer, base_len)` once a row named it.
    base: Option<(u32, u32)>,
    /// Positions of the tail from the base on.
    tail_len: u32,
    /// Where the live image stands, when it holds the state after the slot's last row's
    /// host tokens.
    live_pos: Option<u32>,
    /// Positions the kept normed states hold, row by row.
    hiddens: Vec<u32>,
}

/// One KV group's pool: per block, per layer, keys `[bs, width]` then values, BF16; with
/// drafting, the drafter's group also keeps a tap per block (the target's normed state
/// at the block's last position, BF16).
struct KvPool {
    data: Alloc,
    block_bytes: usize,
    taps: Option<Alloc>,
    tap_bytes: usize,
    layers: u32,
}

impl KvPool {
    /// The block's image, in the order it is zeroed, copied and enciphered.
    fn ranges(&self, b: u32) -> Vec<(u64, usize)> {
        let mut r = vec![(
            self.data.addr + u64::from(b) * self.block_bytes as u64,
            self.block_bytes,
        )];
        if let Some(t) = &self.taps {
            r.push((
                t.addr + u64::from(b) * self.tap_bytes as u64,
                self.tap_bytes,
            ));
        }
        r
    }
}

/// The CUDA executor for Qwen3.8.
pub struct QwenCudaExecutor<D: QwenDevice> {
    dev: D,
    cfg: QwenCudaExecutorConfig,
    spec: ModelSpec,
    dims: Dims,
    weights: DevWeights,
    pools: Vec<KvPool>,
    host_pools: Vec<HostAlloc>,
    /// State buffers: `[num_buffers]` images of `image_stride` bytes (data, then a tag).
    states: Alloc,
    host_states: Option<HostAlloc>,
    image_stride: usize,
    /// Live images, one per slot and the padding slot.
    live: Alloc,
    /// Tails, per DeltaNet layer `[slot][cap]` rows: the raw conv inputs (`conv_dim` f32)
    /// and the `a`, `b` rows (`2 · v_heads` f32).
    tails_conv: Alloc,
    tails_ab: Alloc,
    /// The blocked DeltaNet's tail cache: per DeltaNet layer `[slot][ring]` entries of
    /// a position's intermediates (8 bytes without blocks).
    tails_dn: Alloc,
    tail_cap: u32,
    /// Kept normed states: per slot `[k + 1][hidden]` BF16.
    hiddens: Alloc,
    /// `[slot][group][logical block]`.
    tables: Vec<Vec<Vec<u32>>>,
    slots: Vec<SlotRec>,
    kv_encrypted: Vec<Vec<bool>>,
    state_encrypted: Vec<bool>,
    /// Host mirror of each state buffer's tag (its boundary plus one).
    state_tags: Vec<u32>,
    host_kv_encrypted: Vec<Vec<bool>>,
    host_state_encrypted: Vec<bool>,
    host_state_tags: Vec<u32>,
    steps: u64,
    scratch: Scratch,
    /// The table the program reads: a rung's fixed one, or the eager one.
    table_base: Cell<u64>,
    eager_table: Alloc,
    eager_words: usize,
    rungs: Vec<Rung>,
    path: DecodePath,
    stats: DecodeStats,
    had_rows: bool,
    idle_scrubs: u64,
}

fn ex(e: CudaError) -> ExecutorError {
    ExecutorError(e.to_string())
}

/// A count the configuration bounds far below 2^32 (rows, layers, drafts).
fn n32(x: usize) -> u32 {
    u32::try_from(x).expect("a count within the kernels' 32-bit range")
}

fn u32of(x: usize, what: &str) -> Result<u32> {
    narrow(x, what)
}

impl<D: QwenDevice> QwenCudaExecutor<D> {
    /// An executor over `model` on `dev`. Refuses a configuration the kernels cannot run.
    pub fn new(dev: D, model: &QwenModel, cfg: QwenCudaExecutorConfig) -> Result<Self> {
        let c = model.config();
        let a = &c.attention;
        let d = &c.deltanet;
        if d.k_head_dim != KEY_DIM as usize || d.v_head_dim > 128 || d.conv_kernel != 4 {
            return Err(CudaError::new(
                "the DeltaNet kernels take key dim 128, value dim at most 128, conv width 4",
            ));
        }
        if a.head_dim > 256 || a.head_dim > ROPE_MAX as usize || a.rotary_dim > a.head_dim {
            return Err(CudaError::new("attention head dim past the kernels'"));
        }
        if dev.profile() == QwenProfile::Fast && a.head_dim != super::attention::HEAD_DIM as usize {
            return Err(CudaError::new(
                "the fast profile's attention is compiled for 256-wide heads",
            ));
        }
        if !c.hidden_size.is_multiple_of(128) || !c.intermediate_size.is_multiple_of(128) {
            return Err(CudaError::new("widths must be whole 128-blocks"));
        }
        if c.hidden_size > NORM_MAX as usize {
            return Err(CudaError::new("hidden size past the norm's staged width"));
        }
        let dims = Dims::of(c)?;
        let k = cfg.draft_tokens;
        let drafting = k > 0;
        let dn_layers = dims
            .kinds
            .iter()
            .filter(|&&l| l == LayerKind::DeltaNet)
            .count();
        let attn_layers = dims.kinds.len() - dn_layers;
        let rope_positions = cfg.max_model_len + k + 1;
        let weights = DevWeights::load(&dev, model, drafting, rope_positions)?;
        let bs = cfg.block_size;
        let w = dims.kv_width() as usize;
        let mut pools = Vec::new();
        let groups = if drafting { 2 } else { 1 };
        for g in 0..groups {
            let layers = if g == TARGET { attn_layers } else { 1 };
            let block_bytes = layers * 2 * bs as usize * w * 2;
            // One pad block past the seam's ids, for padding rows.
            let data = Alloc::new(&dev, (cfg.num_blocks as usize + 1) * block_bytes)?;
            let tap_bytes = dims.hidden as usize * 2;
            let taps = if g == DRAFTER {
                Some(Alloc::new(&dev, (cfg.num_blocks as usize + 1) * tap_bytes)?)
            } else {
                None
            };
            pools.push(KvPool {
                data,
                block_bytes,
                taps,
                tap_bytes,
                layers: u32of(layers, "layers")?,
            });
        }
        let tiered = cfg.host_blocks > 0;
        if tiered != (cfg.host_state_buffers > 0) {
            return Err(CudaError::new(
                "a host tier for the KV and the state alike, or neither",
            ));
        }
        let mut host_pools = Vec::new();
        if tiered {
            for p in &pools {
                let bytes = cfg.host_blocks as usize
                    * (p.block_bytes + p.tap_bytes * usize::from(p.taps.is_some()));
                let (ptr, keep) = dev.alloc_host(bytes)?;
                host_pools.push(HostAlloc { ptr, _keep: keep });
            }
        }
        let data_bytes = dn_layers * dims.layer_floats() * 4;
        // Images a whole number of conv rows apart (and of 256 bytes), so the row copies
        // address any image's conv history as rows of one buffer.
        let row = dims.conv_dim() as usize * 4;
        let image_stride = (data_bytes + 4).next_multiple_of(lcm(256, row));
        let states = Alloc::new(&dev, cfg.state_buffers as usize * image_stride)?;
        let host_states = if tiered {
            let (ptr, keep) = dev.alloc_host(cfg.host_state_buffers as usize * image_stride)?;
            Some(HostAlloc { ptr, _keep: keep })
        } else {
            None
        };
        let slots_n = cfg.num_state_slots as usize;
        // The padding slot's state after every real slot's.
        let live = Alloc::new(&dev, (slots_n + 1) * image_stride)?;
        let tail_cap = bs - 1 + k;
        // A ring of at least one row a slot: with one-position blocks and no drafts a
        // slot keeps no tail, and nothing is addressed in it.
        let tail_rows = dn_layers * (slots_n + 1) * tail_cap.max(1) as usize;
        let tails_conv = Alloc::new(&dev, tail_rows * dims.conv_dim() as usize * 4)?;
        let tails_ab = Alloc::new(&dev, tail_rows * 2 * dims.v_heads as usize * 4)?;
        let tails_dn = Alloc::new(
            &dev,
            if dev.block_deltanet() {
                tail_rows * dims.chunk_entry() * 4
            } else {
                8
            },
        )?;
        let hiddens = Alloc::new(
            &dev,
            (slots_n + 1) * (k as usize + 1) * dims.hidden as usize * 2,
        )?;
        let group = |name: &str, role, layers: usize| KvGroupSpec {
            name: name.into(),
            role,
            attention: AttentionKind::Full,
            num_layers: n32(layers),
            num_kv_heads: dims.kv_heads,
            head_dim_qk: dims.head_dim,
            head_dim_v: dims.head_dim,
            num_blocks: cfg.num_blocks,
            host_blocks: cfg.host_blocks,
            ratio: 1,
            device: 0,
            coverage: Coverage::All,
        };
        let mut kv_groups = vec![group("attention", KvRole::Target, attn_layers)];
        if drafting {
            kv_groups.push(group("mtp", KvRole::Drafter, 1));
        }
        let spec = ModelSpec {
            vocab_size: dims.vocab,
            sampleable_vocab_size: cfg.sampleable_vocab_size,
            block_size: bs,
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
                    num_layers: u32of(dn_layers, "layers")?,
                    device: 0,
                    image_bytes: (data_bytes + 4) as u64,
                }],
                num_buffers: cfg.state_buffers,
                host_buffers: cfg.host_state_buffers,
                tail_positions: tail_cap,
            }),
        };
        spec.validate()
            .map_err(|e| CudaError::new(format!("spec: {e:?}")))?;
        let width = spec.max_blocks_per_seq() as usize;
        let caps = capacities(&cfg, tail_cap);
        let k_max = dims.k_max();
        let scratch = Self::alloc_scratch(&dev, &dims, k_max, caps)?;
        let eager_table = Alloc::new(&dev, 64 * 4)?;
        let graphs = cfg.graphs;
        let mut ex = Self {
            tables: vec![vec![vec![NULL_BLOCK; width]; groups]; slots_n],
            slots: vec![SlotRec::default(); slots_n],
            kv_encrypted: vec![vec![false; cfg.num_blocks as usize]; groups],
            state_encrypted: vec![false; cfg.state_buffers as usize],
            state_tags: vec![0; cfg.state_buffers as usize],
            host_kv_encrypted: vec![
                vec![false; cfg.host_blocks as usize];
                if tiered { groups } else { 0 }
            ],
            host_state_encrypted: vec![false; cfg.host_state_buffers as usize],
            host_state_tags: vec![0; cfg.host_state_buffers as usize],
            dev,
            dims,
            weights,
            pools,
            host_pools,
            states,
            host_states,
            image_stride,
            live,
            tails_conv,
            tails_ab,
            tails_dn,
            tail_cap,
            hiddens,
            spec,
            cfg,
            steps: 0,
            table_base: Cell::new(scratch.table.addr),
            scratch,
            eager_table,
            eager_words: 64,
            rungs: Vec::new(),
            path: DecodePath::Eager,
            stats: DecodeStats::default(),
            had_rows: false,
            idle_scrubs: 0,
        };
        ex.prepare_rungs(graphs == CudaGraphs::On)?;
        Ok(ex)
    }

    pub fn device(&self) -> &D {
        &self.dev
    }

    pub fn config(&self) -> &QwenCudaExecutorConfig {
        &self.cfg
    }

    pub fn profile(&self) -> QwenProfile {
        self.dev.profile()
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    fn dn_layers(&self) -> usize {
        self.dims
            .kinds
            .iter()
            .filter(|&&l| l == LayerKind::DeltaNet)
            .count()
    }

    fn state_pool(&self) -> u32 {
        n32(self.pools.len())
    }

    fn data_bytes(&self) -> usize {
        self.dn_layers() * self.dims.layer_floats() * 4
    }

    fn state_addr(&self, b: u32) -> u64 {
        self.states.addr + u64::from(b) * self.image_stride as u64
    }

    fn live_addr(&self, slot: Slot) -> u64 {
        self.live.addr + u64::from(slot) * self.image_stride as u64
    }

    /// Bytes of one layer in an image.
    fn layer_bytes(&self) -> usize {
        self.dims.layer_floats() * 4
    }

    /// A slot's tail rows of every layer: per layer one range of each tail array.
    fn tail_ranges(&self, slot: Slot) -> Vec<(u64, usize)> {
        let cap = self.tail_ring() as usize;
        let layer_rows = self.tail_layer_rows();
        let mut r = Vec::new();
        let mut arrays = vec![
            (&self.tails_conv, self.dims.conv_dim() as usize * 4),
            (&self.tails_ab, 2 * self.dims.v_heads as usize * 4),
        ];
        if self.blocked() {
            arrays.push((&self.tails_dn, self.dims.chunk_entry() * 4));
        }
        for (arr, row) in arrays {
            for l in 0..self.dn_layers() {
                let at = (l * layer_rows + slot as usize * cap) * row;
                r.push((arr.addr + at as u64, cap * row));
            }
        }
        r
    }

    fn hidden_row(&self, slot: Slot, i: usize) -> u64 {
        let k1 = self.cfg.draft_tokens as usize + 1;
        self.hiddens.addr + ((slot as usize * k1 + i) * self.dims.hidden as usize * 2) as u64
    }
}

// ---- memory, maintenance and the host-visible state checks ----

impl<D: QwenDevice> QwenCudaExecutor<D> {
    /// The ranges of a block of `group` (the state pool's index for a state buffer).
    fn block_ranges(&self, group: u32, b: u32) -> Vec<(u64, usize)> {
        if group == self.state_pool() {
            let base = self.state_addr(b);
            return vec![
                (base, self.data_bytes()),
                (base + self.data_bytes() as u64, 4),
            ];
        }
        self.pools[group as usize].ranges(b)
    }

    /// Bytes of a host-tier block of `group`.
    fn host_block_bytes(&self, group: u32) -> usize {
        if group == self.state_pool() {
            return self.image_stride;
        }
        let p = &self.pools[group as usize];
        p.block_bytes + if p.taps.is_some() { p.tap_bytes } else { 0 }
    }

    fn host_block_ptr(&self, group: u32, host: u32) -> *mut u8 {
        let base = if group == self.state_pool() {
            self.host_states.as_ref().expect("a host tier").ptr
        } else {
            self.host_pools[group as usize].ptr
        };
        // SAFETY: inside the host pool (host < host_blocks, checked by the caller).
        unsafe { base.add(host as usize * self.host_block_bytes(group)) }
    }

    fn zero_block(&mut self, group: u32, b: u32) -> Result<()> {
        for (addr, bytes) in self.block_ranges(group, b) {
            // SAFETY: a block's ranges lie inside its pool.
            unsafe { self.dev.zero(addr, bytes)? };
        }
        if group == self.state_pool() {
            self.state_encrypted[b as usize] = false;
            self.state_tags[b as usize] = 0;
        } else {
            self.kv_encrypted[group as usize][b as usize] = false;
        }
        Ok(())
    }

    /// A block's image and its keystream, into `batch`: each range from a fresh keystream
    /// block, as `eidola_engine::kv_cipher` defines a block image. The caller XORs the
    /// batch ([`QwenDevice::cipher_ranges`]) before anything else touches the block.
    fn crypt<'k>(
        &mut self,
        group: u32,
        b: u32,
        key: &'k KvKey,
        nonce: &KvNonce,
        encrypt: bool,
        batch: &mut Vec<CipherRange<'k>>,
    ) -> Result<()> {
        let flag = if group == self.state_pool() {
            &mut self.state_encrypted[b as usize]
        } else {
            &mut self.kv_encrypted[group as usize][b as usize]
        };
        assert_ne!(*flag, encrypt, "group {group} block {b}: crypt twice");
        *flag = encrypt;
        let mut counter = 0u32;
        for (addr, bytes) in self.block_ranges(group, b) {
            let words = u32of(bytes / 4, "cipher words")?;
            batch.push(CipherRange {
                addr,
                words,
                counter,
                key,
                nonce: *nonce,
            });
            counter += words.div_ceil(16);
        }
        Ok(())
    }

    /// XORs every range of `batch` with its keystream in one launch, and empties it.
    fn flush_crypts(
        &self,
        batch: &mut Vec<CipherRange<'_>>,
        blocks: &mut Vec<(u32, u32)>,
    ) -> Result<()> {
        // SAFETY: each range is a piece of one block of a pool; the blocks are distinct
        // (`blocks`), so the ranges do not overlap.
        let r = unsafe { self.dev.cipher_ranges(batch) };
        batch.clear();
        blocks.clear();
        r
    }

    fn reset_slot(&mut self, slot: Slot) -> Result<()> {
        let s = slot as usize;
        for g in &mut self.tables[s] {
            g.fill(NULL_BLOCK);
        }
        self.slots[s] = SlotRec::default();
        let k1 = self.cfg.draft_tokens as usize + 1;
        // SAFETY: the slot's own ranges of the live images, tails and hiddens.
        unsafe {
            self.dev.zero(self.live_addr(slot), self.image_stride)?;
            for (a, n) in self.tail_ranges(slot) {
                self.dev.zero(a, n)?;
            }
            self.dev
                .zero(self.hidden_row(slot, 0), k1 * self.dims.hidden as usize * 2)?;
        }
        Ok(())
    }

    fn set_tag(&mut self, b: u32, len: u32) -> Result<()> {
        self.state_tags[b as usize] = len + 1;
        let at = self.state_addr(b) + self.data_bytes() as u64;
        // SAFETY: the buffer's tag word.
        unsafe { self.dev.upload(at, &(len + 1).to_le_bytes()) }
    }

    /// The step's maintenance in its order. Consecutive encryptions and decryptions run
    /// as one cipher launch over all their blocks' ranges, queued before the next other
    /// operation (which may read or move those blocks) and before any block's second one.
    fn maintain(&mut self, step: &StepInput) -> Result<()> {
        let state = self.state_pool();
        let mut batch: Vec<CipherRange<'_>> = Vec::new();
        let mut blocks: Vec<(u32, u32)> = Vec::new();
        for m in &step.maintenance {
            match *m {
                Maintenance::Encrypt { group, block, .. }
                | Maintenance::Decrypt { group, block, .. } => {
                    if blocks.contains(&(group, block)) {
                        self.flush_crypts(&mut batch, &mut blocks)?;
                    }
                }
                _ => self.flush_crypts(&mut batch, &mut blocks)?,
            }
            match *m {
                Maintenance::SaveState { slot, len, buffer } => {
                    self.save_state(slot, len, buffer)?
                }
                Maintenance::ResetSlot { slot } => self.reset_slot(slot)?,
                Maintenance::Copy { group, src, dst } => {
                    assert_ne!(group, state, "a copy names the state pool");
                    let (s, d) = (self.block_ranges(group, src), self.block_ranges(group, dst));
                    for ((sa, n), (da, _)) in s.into_iter().zip(d) {
                        // SAFETY: two distinct blocks of one pool.
                        unsafe { self.dev.copy(da, sa, n)? };
                    }
                    self.kv_encrypted[group as usize][dst as usize] =
                        self.kv_encrypted[group as usize][src as usize];
                }
                Maintenance::Zero { group, block } => self.zero_block(group, block)?,
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
                    self.crypt(group, block, key, &nonce, encrypt, &mut batch)?;
                    blocks.push((group, block));
                }
                Maintenance::Demote { group, block, host } => {
                    let enc = if group == state {
                        self.state_encrypted[block as usize]
                    } else {
                        self.kv_encrypted[group as usize][block as usize]
                    };
                    assert!(
                        enc,
                        "group {group} block {block}: demote of a plaintext block"
                    );
                    let mut at = self.host_block_ptr(group, host);
                    for (addr, n) in self.block_ranges(group, block) {
                        // SAFETY: the block's ranges, into its host block.
                        unsafe {
                            self.dev.copy_to_host(at, addr, n)?;
                            at = at.add(n);
                        }
                    }
                    if group == state {
                        self.host_state_encrypted[host as usize] = true;
                        self.host_state_tags[host as usize] = self.state_tags[block as usize];
                    } else {
                        self.host_kv_encrypted[group as usize][host as usize] = true;
                    }
                }
                Maintenance::Promote { group, host, block } => {
                    let held = if group == state {
                        self.host_state_encrypted[host as usize]
                    } else {
                        self.host_kv_encrypted[group as usize][host as usize]
                    };
                    assert!(held, "group {group} host {host}: promote of no ciphertext");
                    let mut at = self.host_block_ptr(group, host).cast_const();
                    for (addr, n) in self.block_ranges(group, block) {
                        // SAFETY: as Demote, the other way.
                        unsafe {
                            self.dev.copy_from_host(addr, at, n)?;
                            at = at.add(n);
                        }
                    }
                    if group == state {
                        self.state_encrypted[block as usize] = true;
                        self.state_tags[block as usize] = self.host_state_tags[host as usize];
                    } else {
                        self.kv_encrypted[group as usize][block as usize] = true;
                    }
                }
                Maintenance::ZeroHost { group, host } => {
                    self.dev.synchronize()?;
                    let ptr = self.host_block_ptr(group, host);
                    // SAFETY: the host block; the queue has drained.
                    unsafe { std::ptr::write_bytes(ptr, 0, self.host_block_bytes(group)) };
                    if group == state {
                        self.host_state_encrypted[host as usize] = false;
                        self.host_state_tags[host as usize] = 0;
                    } else {
                        self.host_kv_encrypted[group as usize][host as usize] = false;
                    }
                }
            }
        }
        self.flush_crypts(&mut batch, &mut blocks)
    }
}

// ---- the step's plan ----

/// Where a run's recurrence starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Src {
    /// The zero state.
    Zero,
    /// A slot's live image.
    Live(Slot),
    /// A state buffer.
    Buffer(u32),
}

/// One run through the target (or, with no tokens, a fold alone): its host tokens, then
/// its drafts.
#[derive(Clone, Debug)]
struct Run {
    slot: Slot,
    /// The first new position.
    start: u32,
    /// Host ids.
    host: Vec<u32>,
    /// The drafts after them, by lane index.
    lanes: Vec<u32>,
    src: Src,
    /// The position the source stands at: the kept positions `src_pos .. start` are
    /// folded from the tail first.
    src_pos: u32,
    /// State writes `(len, buffer)`, ascending, at most the host tokens' end.
    writes: Vec<(u32, u32)>,
    /// Store the state after the host tokens into the slot's live image.
    live_store: bool,
    /// New positions at or past this join the tail.
    tail_from: u32,
}

impl Run {
    fn fold(&self) -> u32 {
        self.start - self.src_pos
    }
    fn n_host(&self) -> u32 {
        n32(self.host.len())
    }
    fn n(&self) -> u32 {
        n32(self.host.len() + self.lanes.len())
    }
    fn host_end(&self) -> u32 {
        self.start + self.n_host()
    }
}

/// One row of a step: a real row, or a padding row of a graph rung.
#[derive(Clone, Debug)]
struct Row {
    slot: Slot,
    c: u32,
    toks: Vec<u32>,
    k: u32,
    sample: bool,
    params: SamplingParams,
    run: Run,
    /// The new base `(buffer, len)` after the row's writes.
    new_base: (u32, u32),
    /// The drafter's hidden input for its first host position: a kept row or a tap.
    hidden_before: Option<(u32, u32)>,
}

impl Row {
    fn p(&self) -> u32 {
        self.c + n32(self.toks.len()) - 1
    }
    fn e(&self) -> u32 {
        self.p() + 1
    }
}

/// Where a step's rows sit in its plan, for reading the results.
#[derive(Clone, Debug, Default)]
struct Placed {
    /// Per row: its last host row's head row, its drafting index, its first draft's
    /// index among the head's draft rows (which follow the `hosts` host rows).
    head: Vec<Option<u32>>,
    drafting: Vec<Option<u32>>,
    first_draft: Vec<u32>,
    hosts: u32,
    /// Per row: its index among the plain draws.
    plain: Vec<Option<u32>>,
}

/// A captured decode rung: its rows and draft width, its shape and layout, and its graph.
struct Rung {
    width: u32,
    rows: u32,
    shape: Shape,
    layout: Layout,
    graph: Option<Box<dyn Any>>,
}

/// Steps run by each path ([`QwenCudaExecutor::decode_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeStats {
    pub eager: u64,
    pub direct: u64,
    pub replay: u64,
}

/// The token a padding row embeds.
const PAD_TOKEN: u32 = 0;

impl<D: QwenDevice> QwenCudaExecutor<D> {
    pub(super) fn slot_count(&self) -> usize {
        self.cfg.num_state_slots as usize + 1
    }

    /// The padding rows' slot: past every slot the seam names.
    fn pad_slot(&self) -> Slot {
        self.cfg.num_state_slots
    }

    /// The pad block of each pool: past every block the seam names.
    fn pad_block(&self) -> u32 {
        self.cfg.num_blocks
    }

    /// Whether the DeltaNet runs in blocks (the fast profile's form, `dn_chunk`).
    pub(super) fn blocked(&self) -> bool {
        self.dev.block_deltanet()
    }

    /// Whether every row starts from its base and folds the slot's tail: the lagging
    /// strategy, and always in blocks (a state exists at block boundaries only).
    fn lagging(&self) -> bool {
        self.cfg.strategy == StateStrategy::Lagging || self.blocked()
    }

    /// Positions of a block of the blocked DeltaNet: the block size's largest power-of-two
    /// divisor, at most 16, so every boundary the serving core stores a state at is a
    /// block's.
    pub(super) fn chunk(&self) -> u32 {
        block_chunk(self.cfg.block_size)
    }

    /// Rows of one tail layer: every slot's ring.
    pub(super) fn tail_layer_rows(&self) -> usize {
        self.slot_count() * self.tail_ring() as usize
    }

    /// Rows of a slot's tail ring: its capacity, at least one.
    fn tail_ring(&self) -> u32 {
        self.tail_cap.max(1)
    }

    fn tail_index(&self, slot: Slot, pos: u32) -> u32 {
        slot * self.tail_ring() + pos % self.tail_ring()
    }

    /// Conv rows (of `conv_dim` f32) per image: images are laid out a whole number of
    /// them apart.
    fn rows_per_image(&self) -> u32 {
        n32(self.image_stride / (self.dims.conv_dim() as usize * 4))
    }

    fn src_addr(&self, s: Src) -> u64 {
        match s {
            Src::Zero => 0,
            Src::Live(slot) => self.live_addr(slot),
            Src::Buffer(b) => self.state_addr(b),
        }
    }

    /// Row `j` of a source's conv history at any layer, as a conv-buffer row.
    fn hist_row(&self, s: Src, j: u32) -> (u32, u32) {
        match s {
            Src::Zero => (buf::C_ZERO, j),
            Src::Live(slot) => (buf::C_LIVE, slot * self.rows_per_image() + j),
            Src::Buffer(b) => (buf::C_STATES, b * self.rows_per_image() + j),
        }
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
        assert!(
            !self.kv_encrypted[group][b as usize],
            "slot {slot}: write to an encrypted block"
        );
        b
    }

    /// The block holding `pos` (the pad block for the padding slot).
    fn blk(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        if slot == self.pad_slot() {
            self.pad_block()
        } else {
            self.block_of(slot, group, pos)
        }
    }

    fn wblk(&self, slot: Slot, group: usize, pos: u32) -> u32 {
        if slot == self.pad_slot() {
            self.pad_block()
        } else {
            self.writable_block(slot, group, pos)
        }
    }

    fn table_row(&self, slot: Slot, group: usize) -> Vec<u32> {
        if slot == self.pad_slot() {
            vec![self.pad_block(); self.tables[0][group].len()]
        } else {
            self.tables[slot as usize][group].clone()
        }
    }

    fn geo(&self) -> Geo {
        let g = |page_size| Fa2Geometry {
            group_size: self.dims.q_heads / self.dims.kv_heads,
            page_size,
            partial_rows: n32(self.scratch.caps.partial_rows),
        };
        Geo {
            fast: self.dev.profile() == QwenProfile::Fast,
            table_width: self.tables[0][TARGET].len(),
            target: g(self.cfg.block_size),
            drafter: g(1),
        }
    }

    /// The source a slot's recurrence starts from for positions up to `upto`: its live
    /// image when the strategy keeps one and it stands at or before `upto` past the base,
    /// else its base buffer (the zero state for `NULL_STATE`), and its position.
    fn source(&self, slot: Slot, upto: u32, writes_after: u32) -> (Src, u32) {
        let rec = &self.slots[slot as usize];
        let (base, base_len) = rec.base.expect("a slot with a base");
        if !self.lagging()
            && let Some(lp) = rec.live_pos
            && lp >= base_len
            && lp <= upto
            && lp < writes_after
        {
            return (Src::Live(slot), lp);
        }
        if base == NULL_STATE {
            assert_eq!(
                base_len, 0,
                "the zero state named as the state at {base_len}"
            );
            return (Src::Zero, 0);
        }
        assert!(
            !self.state_encrypted[base as usize],
            "state buffer {base}: read while encrypted"
        );
        assert_eq!(
            self.state_tags[base as usize],
            base_len + 1,
            "state buffer {base}: holds another boundary's state"
        );
        (Src::Buffer(base), base_len)
    }

    /// One target pass over `runs`: its DeltaNet sequences, staging and keeping copies,
    /// and attention rows, the runs' token rows consecutive from 0.
    fn target_pass(&self, runs: &[Run], kv: bool) -> TargetPass {
        let rpi = self.rows_per_image();
        let (blocked, chunk) = (self.blocked(), self.chunk());
        let mut tp = TargetPass::default();
        let (mut t, mut st, mut cv, mut tile) = (0u32, 0u32, 0u32, 0u32);
        for r in runs {
            let (f, nh, n) = (r.fold(), r.n_host(), r.n());
            assert!(
                r.writes.len() <= SEQ_WRITES,
                "more state writes in one row than the kernel takes"
            );
            if blocked {
                // Blocks stand at absolute positions: a run starts at one's boundary (the
                // padding row's position is never read) and stores whole blocks only.
                assert!(
                    r.slot == self.pad_slot() || r.src_pos.is_multiple_of(chunk),
                    "slot {}: a blocked run off a block boundary",
                    r.slot
                );
                assert!(
                    !r.live_store || (r.host_end() - r.src_pos).is_multiple_of(chunk),
                    "slot {}: a live state inside a block",
                    r.slot
                );
                assert!(
                    r.writes
                        .iter()
                        .all(|&(len, _)| (len - r.src_pos).is_multiple_of(chunk)),
                    "slot {}: a state write inside a block",
                    r.slot
                );
            }
            if blocked {
                let (cvn, stn) = self.blocked_staging(r, &mut tp, t, st, cv, tile);
                tile += (f + n).div_ceil(chunk);
                for j in 0..n {
                    let pos = r.start + j;
                    if pos >= r.tail_from {
                        let ti = self.tail_index(r.slot, pos);
                        tp.post_conv
                            .push(Item::new(buf::C_CONV, cv + 3 + j, buf::C_TAIL, ti));
                        tp.post_ab
                            .push(Item::new(buf::A_STAGED, st + j, buf::A_TAIL, ti));
                    }
                    tp.o_rows
                        .push(Item::new(buf::O_STAGED, st + j, buf::O_ROWS, t + j));
                }
                self.attention_rows(r, &mut tp, t, kv);
                t += n;
                st = stn;
                cv = cvn;
                continue;
            }
            let mut seq = DnSeq {
                src: self.src_addr(r.src),
                live: if r.live_store {
                    self.live_addr(r.slot)
                } else {
                    0
                },
                first: st,
                count: f + n,
                out_from: f,
                live_after: if r.live_store { f + nh } else { f + n + 1 },
                writes: n32(r.writes.len()),
                tile,
                ..DnSeq::default()
            };
            for (j, &(len, b)) in r.writes.iter().enumerate() {
                seq.write_after[j] = len - r.src_pos;
                seq.write_dst[j] = self.state_addr(b);
            }
            tp.seqs.push(seq);
            tp.window.extend((0..f + n).map(|j| cv + j));
            for j in 0..3 {
                let (b, row) = self.hist_row(r.src, j);
                tp.stage_conv.push(Item::new(b, row, buf::C_CONV, cv + j));
            }
            for j in 0..f {
                let ti = self.tail_index(r.slot, r.src_pos + j);
                tp.stage_conv
                    .push(Item::new(buf::C_TAIL, ti, buf::C_CONV, cv + 3 + j));
                tp.stage_ab
                    .push(Item::new(buf::A_TAIL, ti, buf::A_STAGED, st + j));
            }
            for j in 0..n {
                tp.stage_conv
                    .push(Item::new(buf::C_NEW, t + j, buf::C_CONV, cv + 3 + f + j));
                tp.stage_ab
                    .push(Item::new(buf::A_NEW, t + j, buf::A_STAGED, st + f + j));
            }
            // The conv history after `len`: positions `len - 3 .. len`.
            let hist = |len: u32| cv + (len - r.src_pos);
            for &(len, b) in &r.writes {
                for j in 0..3 {
                    tp.post_conv.push(Item::new(
                        buf::C_CONV,
                        hist(len) + j,
                        buf::C_STATES,
                        b * rpi + j,
                    ));
                }
            }
            if r.live_store {
                for j in 0..3 {
                    tp.post_conv.push(Item::new(
                        buf::C_CONV,
                        hist(r.host_end()) + j,
                        buf::C_LIVE,
                        r.slot * rpi + j,
                    ));
                }
            }
            for j in 0..n {
                let pos = r.start + j;
                if pos >= r.tail_from {
                    let ti = self.tail_index(r.slot, pos);
                    tp.post_conv
                        .push(Item::new(buf::C_CONV, cv + 3 + f + j, buf::C_TAIL, ti));
                    tp.post_ab
                        .push(Item::new(buf::A_STAGED, st + f + j, buf::A_TAIL, ti));
                }
                tp.o_rows
                    .push(Item::new(buf::O_STAGED, st + f + j, buf::O_ROWS, t + j));
            }
            self.attention_rows(r, &mut tp, t, kv);
            t += n;
            st += f + n;
            cv += 3 + f + n;
        }
        tp.conv_rows = cv;
        tp.tiles = tile;
        tp
    }

    /// A run's token rows, draft copies and attention rows, its first token row `t`.
    fn attention_rows(&self, r: &Run, tp: &mut TargetPass, t: u32, kv: bool) {
        let bs = self.cfg.block_size;
        let fast = self.dev.profile() == QwenProfile::Fast;
        let (nh, n) = (r.n_host(), r.n());
        for j in 0..n {
            let pos = r.start + j;
            tp.attn.rope.push(pos);
            tp.attn.query.push(pos);
            tp.attn.targets.push(if kv {
                self.wblk(r.slot, TARGET, pos) * bs + pos % bs
            } else {
                NO_TARGET
            });
            if !fast {
                tp.attn.tables.extend(self.table_row(r.slot, TARGET));
            }
        }
        tp.tokens.extend_from_slice(&r.host);
        tp.tokens.extend(r.lanes.iter().map(|_| PAD_TOKEN));
        tp.token_copies.extend(
            r.lanes
                .iter()
                .enumerate()
                .map(|(j, &lane)| Item::new(buf::T_LANES, lane, buf::T_DST, t + nh + n32(j))),
        );
        if fast && n > 0 {
            tp.attn
                .requests
                .push(plan::request(t, n, r.start + n - 1, bs, &mut |j| {
                    self.blk(r.slot, TARGET, j * bs)
                }));
        }
    }

    /// The raw convolution input row of `r`'s slot at position `pos` before the run's new
    /// positions (`pos < r.start`): from the source's history before its position, else
    /// from the tail. `None` before the sequence's start.
    fn raw_row_before(&self, r: &Run, pos: i64) -> (u32, u32) {
        let src_pos = i64::from(r.src_pos);
        if pos < src_pos {
            // The history holds the three positions before the source's.
            let j = u32::try_from(pos - (src_pos - 3)).expect("a history row");
            self.hist_row(r.src, j)
        } else {
            let p = u32::try_from(pos).expect("a position");
            (buf::C_TAIL, self.tail_index(r.slot, p))
        }
    }

    /// The blocked DeltaNet's staging of run `r` (its first token row `t`, staged
    /// position `st`, conv row `cv` and block `tile`): its sequence; the three raw rows
    /// before its new positions, then the new rows, as conv rows; the new positions'
    /// `a`/`b` rows; and the convolution histories its state writes and live store take.
    /// The positions it keeps from earlier rows are read from the tail cache, not staged.
    /// Returns the conv row and staged position after it.
    fn blocked_staging(
        &self,
        r: &Run,
        tp: &mut TargetPass,
        t: u32,
        st: u32,
        cv: u32,
        tile: u32,
    ) -> (u32, u32) {
        let rpi = self.rows_per_image();
        let (f, nh, n) = (r.fold(), r.n_host(), r.n());
        let mut seq = DnSeq {
            src: self.src_addr(r.src),
            live: if r.live_store {
                self.live_addr(r.slot)
            } else {
                0
            },
            first: st,
            count: f + n,
            out_from: f,
            live_after: if r.live_store { f + nh } else { f + n + 1 },
            writes: n32(r.writes.len()),
            tile,
            pos0: r.src_pos,
            tail_row: r.slot * self.tail_ring(),
            keep_from: if r.tail_from == u32::MAX {
                u32::MAX
            } else {
                r.tail_from - r.src_pos
            },
            ..DnSeq::default()
        };
        for (j, &(len, b)) in r.writes.iter().enumerate() {
            seq.write_after[j] = len - r.src_pos;
            seq.write_dst[j] = self.state_addr(b);
        }
        tp.seqs.push(seq);
        tp.window.extend((0..n).map(|j| cv + j));
        // Conv row `cv + i` holds position `start − 3 + i`.
        let first = i64::from(r.start) - 3;
        for j in 0..3u32 {
            let (b, row) = self.raw_row_before(r, first + i64::from(j));
            tp.stage_conv.push(Item::new(b, row, buf::C_CONV, cv + j));
        }
        for j in 0..n {
            tp.stage_conv
                .push(Item::new(buf::C_NEW, t + j, buf::C_CONV, cv + 3 + j));
            tp.stage_ab
                .push(Item::new(buf::A_NEW, t + j, buf::A_STAGED, st + j));
        }
        // The conv history after `len`: positions `len − 3 .. len`, staged from
        // `start − 3` on and read where they are before.
        let history = |len: u32, j: u32| -> (u32, u32) {
            let pos = i64::from(len) - 3 + i64::from(j);
            if pos >= first {
                (
                    buf::C_CONV,
                    cv + u32::try_from(pos - first).expect("a staged row"),
                )
            } else {
                self.raw_row_before(r, pos)
            }
        };
        for &(len, b) in &r.writes {
            for j in 0..3 {
                let (sb, sr) = history(len, j);
                tp.post_conv
                    .push(Item::new(sb, sr, buf::C_STATES, b * rpi + j));
            }
        }
        if r.live_store {
            for j in 0..3 {
                let (sb, sr) = history(r.host_end(), j);
                tp.post_conv
                    .push(Item::new(sb, sr, buf::C_LIVE, r.slot * rpi + j));
            }
        }
        (cv + 3 + n, st + n)
    }

    /// The drafter's attention rows for rows at positions `s` of `slot`s, consecutive
    /// positions of one slot sharing a request.
    fn drafter_attn(&self, rows: &[(Slot, u32)]) -> AttnRows {
        let bs = self.cfg.block_size;
        let fast = self.dev.profile() == QwenProfile::Fast;
        let mut a = AttnRows::default();
        let mut i = 0;
        while i < rows.len() {
            let (slot, s0) = rows[i];
            let mut j = i + 1;
            while j < rows.len() && rows[j].0 == slot && rows[j].1 == rows[j - 1].1 + 1 {
                j += 1;
            }
            for &(_, s) in &rows[i..j] {
                a.rope.push(s - 1);
                a.query.push(s);
                a.targets.push(self.wblk(slot, DRAFTER, s) * bs + s % bs);
                if !fast {
                    a.tables.extend(self.table_row(slot, DRAFTER));
                }
            }
            if fast {
                let last = rows[j - 1].1;
                // Keys from position 1: relative key `x` is position `x + 1`.
                a.requests
                    .push(plan::request(n32(i), n32(j - i), last - 1, 1, &mut |x| {
                        let pos = x + 1;
                        let page = super::attention::drafter_page(
                            self.blk(slot, DRAFTER, pos),
                            pos % bs,
                            bs,
                        );
                        u32::try_from(page).expect("a page id within the pool")
                    }));
            }
            let _ = s0;
            i = j;
        }
        a
    }

    /// The step's plan over `rows` (real rows, then any padding rows).
    fn plan(&self, rows: &[Row], drafting_exec: bool) -> (StepPlan, Placed) {
        let bs = self.cfg.block_size;
        let k1 = self.cfg.draft_tokens + 1;
        let width = rows.iter().map(|r| r.k).max().unwrap_or(0);
        let mut pl = Placed::default();
        let mut p = StepPlan {
            width,
            ..StepPlan::default()
        };
        let draft_rows: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].k > 0).collect();
        let dn = n32(draft_rows.len());
        p.drafting = dn;
        pl.drafting = vec![None; rows.len()];
        for (d, &i) in draft_rows.iter().enumerate() {
            pl.drafting[i] = Some(n32(d));
        }
        // The target's rows: each row's host tokens, then its drafts from their lanes.
        let mut t_first = Vec::with_capacity(rows.len());
        let mut t = 0;
        let runs: Vec<Run> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                t_first.push(t);
                let mut run = r.run.clone();
                if let Some(d) = pl.drafting[i] {
                    run.lanes = (0..r.k).map(|j| j * dn + d).collect();
                }
                t += run.n();
                run
            })
            .collect();
        // The target's normed row of row `i`'s position `q` (a host position or a draft's).
        let normed =
            |i: usize, q: u32| -> (u32, u32) { (buf::H_NORMED, t_first[i] + q - rows[i].c) };
        if drafting_exec && dn > 0 {
            // The drafter at each drafting row's one host position `p`: the token at `p`
            // and the target's normed state at `p - 1`, both from earlier steps.
            let mut pass = DrafterPass::default();
            let mut at = Vec::new();
            for (d, &i) in draft_rows.iter().enumerate() {
                let r = &rows[i];
                let m = n32(d);
                pass.token_copies
                    .push(Item::new(buf::T_HOST, t_first[i], buf::T_DST, m));
                let (hb, hr) = r.hidden_before.expect("a hidden state before the row");
                pass.hidden_copies.push(Item::new(hb, hr, buf::H_DST, m));
                pass.head.push(Item::new(buf::H_DRAFT, m, buf::H_DST, m));
                pass.draws.push(SampleRow::new(&r.params, r.p() + 1, m));
                at.push((r.slot, r.p()));
            }
            pass.attn = self.drafter_attn(&at);
            p.lead.push(pass);
            // The chain: depth `j` drafts from the previous depth's normed rows.
            let mut prev: Vec<u32> = (0..dn).collect();
            for j in 1..width {
                let mut pass = DrafterPass::default();
                let mut at = Vec::new();
                let mut now = vec![0u32; draft_rows.len()];
                for (d, &i) in draft_rows.iter().enumerate() {
                    let r = &rows[i];
                    if r.k > j {
                        let m = n32(at.len());
                        pass.token_copies.push(Item::new(
                            buf::T_LANES,
                            (j - 1) * dn + n32(d),
                            buf::T_DST,
                            m,
                        ));
                        pass.hidden_copies
                            .push(Item::new(buf::H_DRAFT, prev[d], buf::H_DST, m));
                        pass.head.push(Item::new(buf::H_DRAFT, m, buf::H_DST, m));
                        now[d] = m;
                        at.push((r.slot, r.p() + j));
                    }
                }
                pass.attn = self.drafter_attn(&at);
                for (d, &i) in draft_rows.iter().enumerate() {
                    let r = &rows[i];
                    let row = if r.k > j { now[d] } else { 0 };
                    pass.draws
                        .push(SampleRow::new(&r.params, r.p() + 1 + j, row));
                }
                prev = now;
                p.lead.push(pass);
            }
        }
        p.target = self.target_pass(&runs, true);
        // The head: every sampled row's last host row, then every draft's row.
        pl.head = vec![None; rows.len()];
        for (i, r) in rows.iter().enumerate() {
            if r.sample {
                pl.head[i] = Some(n32(p.head.len()));
                let (hb, hr) = normed(i, r.p());
                p.head
                    .push(Item::new(hb, hr, buf::H_DST, n32(p.head.len())));
            }
        }
        let hosts = n32(p.head.len());
        pl.hosts = hosts;
        pl.first_draft = vec![0; rows.len()];
        let mut nd = 0;
        for &i in &draft_rows {
            let r = &rows[i];
            pl.first_draft[i] = nd;
            for j in 0..r.k {
                let (hb, hr) = normed(i, r.e() + j);
                p.head.push(Item::new(hb, hr, buf::H_DST, hosts + nd + j));
            }
            nd += r.k;
        }
        if drafting_exec {
            // The drafter's KV at the host positions of rows that do not draft: from the
            // target's normed rows of this step past the row's first position.
            let mut pass = DrafterPass {
                kv_only: true,
                ..DrafterPass::default()
            };
            let mut at = Vec::new();
            for (i, r) in rows.iter().enumerate() {
                if r.k > 0 {
                    continue;
                }
                for s in r.c.max(1)..=r.p() {
                    let m = n32(at.len());
                    pass.token_copies.push(Item::new(
                        buf::T_HOST,
                        t_first[i] + s - r.c,
                        buf::T_DST,
                        m,
                    ));
                    let (hb, hr) = if s == r.c {
                        r.hidden_before.expect("a hidden state before the row")
                    } else {
                        normed(i, s - 1)
                    };
                    pass.hidden_copies.push(Item::new(hb, hr, buf::H_DST, m));
                    at.push((r.slot, s));
                }
            }
            if !at.is_empty() {
                pass.attn = self.drafter_attn(&at);
                p.trail = Some(pass);
            }
        }
        // Draws, target distributions, acceptance.
        pl.plain = vec![None; rows.len()];
        for (i, r) in rows.iter().enumerate() {
            if r.sample && r.k == 0 {
                pl.plain[i] = Some(n32(p.plain.len()));
                p.plain.push(SampleRow::new(
                    &r.params,
                    r.p() + 1,
                    pl.head[i].expect("a sampled row's head row"),
                ));
            }
        }
        for &i in &draft_rows {
            let r = &rows[i];
            let first = n32(p.targets.len());
            p.targets.push(SampleRow::new(
                &r.params,
                r.p() + 1,
                pl.head[i].expect("a drafting row samples"),
            ));
            for j in 0..r.k {
                p.targets.push(SampleRow::new(
                    &r.params,
                    r.p() + 1,
                    hosts + pl.first_draft[i] + j,
                ));
            }
            p.accept
                .push((SampleRow::new(&r.params, r.p() + 1, 0), first, r.k));
        }
        if drafting_exec {
            // The canonical rewrite at every draft's position.
            if dn > 0 {
                let mut pass = DrafterPass {
                    kv_only: true,
                    ..DrafterPass::default()
                };
                let mut at = Vec::new();
                for (d, &i) in draft_rows.iter().enumerate() {
                    let r = &rows[i];
                    for j in 0..r.k {
                        let m = n32(at.len());
                        pass.token_copies.push(Item::new(
                            buf::T_LANES,
                            j * dn + n32(d),
                            buf::T_DST,
                            m,
                        ));
                        let (hb, hr) = normed(i, r.p() + j);
                        pass.hidden_copies.push(Item::new(hb, hr, buf::H_DST, m));
                        at.push((r.slot, r.e() + j));
                    }
                }
                pass.attn = self.drafter_attn(&at);
                p.rewrite = Some(pass);
            }
            // Taps at every block's last position, and the kept normed states (none for
            // padding rows, which keep nothing).
            for (i, r) in rows.iter().enumerate() {
                if r.slot == self.pad_slot() {
                    continue;
                }
                for q in r.c..=r.p() + r.k {
                    if (q + 1).is_multiple_of(bs) {
                        let (hb, hr) = normed(i, q);
                        p.keep.push(Item::new(
                            hb,
                            hr,
                            buf::H_TAPS,
                            self.wblk(r.slot, DRAFTER, q),
                        ));
                    }
                }
                for j in 0..=r.k {
                    let (hb, hr) = normed(i, r.p() + j);
                    p.keep.push(Item::new(hb, hr, buf::H_KEPT, r.slot * k1 + j));
                }
            }
        }
        (p, pl)
    }

    /// A padding row of a rung of draft width `w`: the pad slot at position `K`, every
    /// read and write in the pad block and the pad slot's own state.
    fn pad_row(&self, w: u32) -> Row {
        let k = self.cfg.draft_tokens;
        let slot = self.pad_slot();
        Row {
            slot,
            c: k,
            toks: vec![PAD_TOKEN],
            k: w,
            sample: true,
            params: SamplingParams::greedy(),
            run: Run {
                slot,
                start: k,
                host: vec![PAD_TOKEN],
                lanes: Vec::new(),
                src: Src::Zero,
                src_pos: k,
                writes: Vec::new(),
                live_store: false,
                tail_from: u32::MAX,
            },
            new_base: (NULL_STATE, k),
            hidden_before: (k > 0).then_some((buf::H_KEPT, slot * (k + 1))),
        }
    }

    /// The shape of a decode rung of `rows` rows at draft width `w`: an all-padding
    /// step's, with room for the variable lists of any step that fits it.
    fn rung_shape(&self, rows: u32, w: u32) -> Result<Shape> {
        let pads: Vec<Row> = (0..rows).map(|_| self.pad_row(w)).collect();
        let drafting_exec = self.cfg.draft_tokens > 0;
        let (p, _) = self.plan(&pads, drafting_exec);
        let geo = self.geo();
        let mut s = Shape::of(&p, &geo)?;
        let r = rows as usize;
        let bs = self.cfg.block_size as usize;
        let fold = if self.lagging() {
            self.tail_cap as usize
        } else {
            self.cfg.draft_tokens as usize
        };
        let writes = ((fold + 1) / bs + 1).min(SEQ_WRITES);
        let wu = w as usize;
        // A row: its folds, its host token and its drafts staged (in blocks only the
        // host token and the drafts: the folded positions' intermediates are in the
        // tail cache, the blocks they span counted); the state writes, the live image
        // and every new position into the tail kept.
        let t = &mut s.target;
        t.tiles = r * (fold + 1 + wu).div_ceil(self.chunk() as usize);
        t.staged = if self.blocked() {
            r * (1 + wu)
        } else {
            r * (fold + 1 + wu)
        };
        t.conv_rows = 3 * r + t.staged;
        t.stage_conv = 3 * r + t.staged;
        t.stage_ab = t.staged;
        t.post_conv = 3 * writes * r + 3 * r + r * (1 + wu);
        t.post_ab = r * (1 + wu);
        if drafting_exec {
            s.keep = r * (wu + 1) + r * ((wu + 1) / bs + 1);
        }
        if geo.fast {
            let len = self.cfg.max_model_len + self.cfg.draft_tokens + 1;
            let blocks = self.spec.max_blocks_per_seq();
            let caps = |queries: u32, g: &Fa2Geometry, pages: u32| {
                PassCaps::with_split_keys(
                    Reduction::Split,
                    rows,
                    queries,
                    g.group_size,
                    g.page_size,
                    pages,
                    len,
                    super::attention::MAX_TILE,
                    plan::SPLIT_KEYS,
                )
            };
            s.target.attn.fa2 = vec![plan::Fa2Pass {
                caps: caps(1 + w, &geo.target, blocks)?,
                out_row: 0,
            }];
            for d in &mut s.lead {
                d.attn.fa2 = vec![plan::Fa2Pass {
                    caps: caps(1, &geo.drafter, len)?,
                    out_row: 0,
                }];
            }
        }
        Ok(s)
    }
}

/// A slice's bytes.
fn bytes_of<T: Copy>(x: &[T]) -> &[u8] {
    // SAFETY: any initialized `Copy` value's bytes may be read; the ABI structs are
    // `repr(C)` with no padding that carries data the kernels read.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}

// ---- a step ----

impl<D: QwenDevice> QwenCudaExecutor<D> {
    fn row_of(&self, step: &StepInput, i: usize, e: &SeqEntry, writes: Vec<StateWrite>) -> Row {
        assert!(e.num_tokens >= 1, "a row needs at least one host token");
        let start = e.token_start as usize;
        let end = start + e.num_tokens as usize;
        for (j, &pos) in step.positions[start..end].iter().enumerate() {
            assert_eq!(
                pos,
                e.context_len + n32(j),
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
            "slot {}: drafts requested at position 0",
            e.slot
        );
        assert!(
            k == 0 || e.num_tokens == 1,
            "slot {}: drafts on a row of {} host tokens",
            e.slot,
            e.num_tokens
        );
        let base = step.aux.state_rows[i];
        let toks = step.token_ids[start..end].to_vec();
        Row {
            slot: e.slot,
            c: e.context_len,
            k,
            sample: e.sample,
            params: e.sampling,
            run: Run {
                slot: e.slot,
                start: e.context_len,
                host: toks.clone(),
                lanes: Vec::new(),
                src: Src::Zero,
                src_pos: 0,
                writes: writes.iter().map(|w| (w.len, w.buffer)).collect(),
                live_store: !self.blocked(),
                tail_from: 0,
            },
            toks,
            new_base: writes
                .last()
                .map_or((base.base, base.base_len), |w| (w.buffer, w.len)),
            hidden_before: None,
        }
    }

    /// Checks a row's base and writes as the reference executor does, starts a fresh
    /// slot at its base, and fixes the row's source and drafter input.
    fn check_row(&mut self, row: &mut Row, base: StateRow) {
        let slot = row.slot;
        let bs = self.cfg.block_size;
        let StateRow { base, base_len } = base;
        let rec = &mut self.slots[slot as usize];
        match rec.base {
            Some(b) => assert_eq!(
                b,
                (base, base_len),
                "slot {slot}: the row's base is not the slot's"
            ),
            None => {
                assert_eq!(
                    row.c, base_len,
                    "slot {slot}: a fresh slot starts at its base"
                );
                rec.base = Some((base, base_len));
                rec.tail_len = 0;
                rec.live_pos = None;
            }
        }
        assert!(
            base_len.is_multiple_of(bs) && base_len <= row.c,
            "slot {slot}: base off a boundary"
        );
        let writes = &row.run.writes;
        for (i, &(len, buffer)) in writes.iter().enumerate() {
            assert!(
                len > base_len && len <= row.e() && len.is_multiple_of(bs),
                "slot {slot}: a state write at {len} outside ({base_len}, {}]",
                row.e()
            );
            assert!(i == 0 || writes[i - 1].0 < len, "state writes out of order");
            assert!(
                buffer != NULL_STATE && buffer < self.cfg.state_buffers,
                "a state write into buffer {buffer}"
            );
            if buffer == base {
                assert_eq!(
                    i + 1,
                    writes.len(),
                    "slot {slot}: its base overwritten early"
                );
            } else {
                assert!(
                    self.state_tags[buffer as usize] == 0 && !self.state_encrypted[buffer as usize],
                    "state buffer {buffer}: written while it holds a state"
                );
            }
        }
        let first_write = writes.first().map_or(u32::MAX, |w| w.0);
        let (src, src_pos) = self.source(slot, row.c, first_write);
        row.run.src = src;
        row.run.src_pos = src_pos;
        row.run.tail_from = row.new_base.1;
        if self.cfg.draft_tokens > 0 && row.c >= 1 {
            row.hidden_before = Some(self.hidden_before(slot, row.c - 1));
        }
    }

    /// The target's normed state at `pos` for `slot`: what its last row kept, or the tap
    /// of the drafter block ending there (after a resume).
    fn hidden_before(&self, slot: Slot, pos: u32) -> (u32, u32) {
        let k1 = self.cfg.draft_tokens + 1;
        if let Some(i) = self.slots[slot as usize]
            .hiddens
            .iter()
            .position(|&q| q == pos)
        {
            return (buf::H_KEPT, slot * k1 + n32(i));
        }
        (buf::H_TAPS, self.block_of(slot, DRAFTER, pos))
    }

    /// Uploads `words` as the table the program reads: the rung's fixed table, or the
    /// eager table (grown as needed).
    fn upload_table(&mut self, words: &[u32], fixed: bool) -> Result<()> {
        let addr = if fixed {
            self.scratch.table.addr
        } else {
            if self.eager_words < words.len() {
                let n = words.len().next_power_of_two();
                self.eager_table = Alloc::new(&self.dev, n * 4)?;
                self.eager_words = n;
            }
            self.eager_table.addr
        };
        self.table_base.set(addr);
        // SAFETY: the table holds `words`.
        unsafe { self.dev.upload(addr, bytes_of(words)) }
    }

    fn readback(&self, words: usize) -> Result<Vec<u32>> {
        let mut out = vec![0u8; words * 4];
        // SAFETY: the read-back holds the capacity's words.
        unsafe { self.dev.download(self.scratch.readback.addr, &mut out)? };
        Ok(out
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// The rung a step of `rows` uniform decode rows at draft width `w` replays, if any.
    fn rung_for(&self, rows: &[Row]) -> Option<usize> {
        let drafting = self.cfg.draft_tokens > 0;
        let w = rows.first()?.k;
        let uniform = rows
            .iter()
            .all(|r| r.toks.len() == 1 && r.sample && r.k == w && (!drafting || r.c >= 1));
        if !uniform {
            return None;
        }
        self.rungs
            .iter()
            .enumerate()
            .filter(|(_, g)| g.width == w && g.rows as usize >= rows.len())
            .min_by_key(|(_, g)| g.rows)
            .map(|(i, _)| i)
    }

    fn step(&mut self, step: &StepInput) -> Result<StepOutput> {
        self.maintain(step)?;
        for u in &step.table_updates {
            assert!(
                (u.group as usize) < self.pools.len(),
                "a table update names group {}",
                u.group
            );
            self.tables[u.slot as usize][u.group as usize][u.index as usize] = u.block;
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
            rows.push(self.row_of(step, i, e, by_row.remove(&n32(i)).unwrap_or_default()));
        }
        for (i, row) in rows.iter_mut().enumerate() {
            self.check_row(row, step.aux.state_rows[i]);
        }
        let real = rows.len();
        if real == 0 {
            if std::mem::take(&mut self.had_rows) {
                self.idle_scrub()?;
            }
            return Ok(StepOutput {
                tokens: Vec::new(),
                stride: self.spec.max_draft_tokens + 1,
                num_tokens: Vec::new(),
                logits: step.return_logits.then(Vec::new),
            });
        }
        self.had_rows = true;
        let drafting_exec = self.cfg.draft_tokens > 0;
        let geo = self.geo();

        // A rung's padded step when one fits, else the step's own.
        let mut ran: Option<(Layout, Placed)> = None;
        if self.path != DecodePath::Eager
            && let Some(g) = self.rung_for(&rows)
        {
            let (width, rung_rows) = (self.rungs[g].width, self.rungs[g].rows);
            let mut padded = rows.clone();
            padded.extend((real..rung_rows as usize).map(|_| self.pad_row(width)));
            let (p, placed) = self.plan(&padded, drafting_exec);
            if let Ok(words) = plan::pack(&p, &self.rungs[g].layout, &geo) {
                self.upload_table(&words, true)?;
                if self.path == DecodePath::Replay {
                    let graph = self.rungs[g]
                        .graph
                        .as_ref()
                        .ok_or_else(|| CudaError::new("no graph captured for this rung"))?;
                    self.dev.replay(graph.as_ref())?;
                    self.stats.replay += 1;
                } else {
                    self.program(&self.rungs[g].layout.clone())?;
                    self.stats.direct += 1;
                }
                ran = Some((self.rungs[g].layout.clone(), placed));
            }
        }
        let (layout, placed) = match ran {
            Some(x) => x,
            None => {
                let (p, placed) = self.plan(&rows, drafting_exec);
                let layout = Layout::new(&Shape::of(&p, &geo)?);
                let words = plan::pack(&p, &layout, &geo)?;
                self.upload_table(&words, false)?;
                self.program(&layout)?;
                self.stats.eager += 1;
                (layout, placed)
            }
        };
        let rb = self.readback(layout.readback_words())?;
        if rb[0] != 0 {
            return Err(CudaError::new(format!("step status {:#x}", rb[0])));
        }
        let plain_n = layout.plain.len / SAMPLE_WORDS;
        let k1 = layout.width as usize + 1;
        let out_at = 1 + plain_n;
        let counts_at = out_at + layout.accepts * k1;
        for r in &rows {
            for &(len, b) in &r.run.writes {
                self.set_tag(b, len)?;
            }
        }

        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; real * stride as usize],
            stride,
            num_tokens: vec![0; real],
            logits: step.return_logits.then(Vec::new),
        };
        let vocab = self.dims.vocab as usize;
        for (i, r) in rows.iter().enumerate() {
            let produced: Vec<u32> = if !r.sample {
                Vec::new()
            } else if let Some(d) = placed.drafting[i] {
                let d = d as usize;
                let n = rb[counts_at + d] as usize;
                assert!(n >= 1 && n <= r.k as usize + 1, "accepted {n} tokens");
                rb[out_at + d * k1..out_at + d * k1 + n].to_vec()
            } else {
                vec![rb[1 + placed.plain[i].expect("a plain row") as usize]]
            };
            let base = i * stride as usize;
            out.tokens[base..base + produced.len()].copy_from_slice(&produced);
            out.num_tokens[i] = n32(produced.len());
            if let Some(l) = out.logits.as_mut() {
                let mut rows_l = Vec::new();
                for j in 0..produced.len() {
                    let lr = if j == 0 {
                        placed.head[i].expect("a sampled row") as usize
                    } else {
                        (placed.hosts + placed.first_draft[i]) as usize + j - 1
                    };
                    let mut v = vec![0f32; vocab];
                    let at = self.scratch.logits_t.addr + (lr * vocab * 4) as u64;
                    // SAFETY: a logit row the head wrote.
                    unsafe {
                        let dst =
                            std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), vocab * 4);
                        self.dev.download(at, dst)?;
                    }
                    rows_l.push(v);
                }
                l.push(rows_l);
            }
        }

        let live = !self.lagging();
        for r in &rows {
            let (nb, nl) = r.new_base;
            let rec = &mut self.slots[r.slot as usize];
            rec.base = Some((nb, nl));
            rec.tail_len = r.e() + r.k - nl;
            assert!(
                rec.tail_len <= self.tail_cap,
                "slot {}: a tail past its capacity",
                r.slot
            );
            rec.live_pos = live.then_some(r.e());
            if drafting_exec {
                rec.hiddens = (r.p()..=r.p() + r.k).collect();
            }
        }
        Ok(out)
    }

    /// Zeroes every buffer a step leaves its working data in, in place (the graphs
    /// captured these addresses): the program's scratch and tables, each pool's pad
    /// block and the padding slot's state. Nothing a real row computes reads their
    /// earlier contents. Run by the first rowless step after a step with rows.
    fn idle_scrub(&mut self) -> Result<()> {
        let mut ranges = self.scratch_ranges();
        ranges.push((self.eager_table.addr, self.eager_words * 4));
        let pad = self.pad_block();
        for g in 0..self.pools.len() {
            ranges.extend(self.pools[g].ranges(pad));
        }
        let ps = self.pad_slot();
        ranges.push((self.live_addr(ps), self.image_stride));
        ranges.extend(self.tail_ranges(ps));
        let k1 = self.cfg.draft_tokens as usize + 1;
        ranges.push((self.hidden_row(ps, 0), k1 * self.dims.hidden as usize * 2));
        for (a, n) in ranges {
            // SAFETY: ranges of live allocations of this executor.
            unsafe { self.dev.zero(a, n)? };
        }
        self.idle_scrubs += 1;
        self.dev.synchronize()
    }

    /// Idle scrubs run so far.
    pub fn idle_scrubs(&self) -> u64 {
        self.idle_scrubs
    }

    /// Steps run by each path so far.
    pub fn decode_stats(&self) -> DecodeStats {
        self.stats
    }

    /// How decode steps that fit a rung run: `Replay` needs the graphs captured.
    pub fn set_decode_path(&mut self, path: DecodePath) -> Result<()> {
        if path == DecodePath::Replay && self.rungs.iter().any(|r| r.graph.is_none()) {
            return Err(CudaError::new("replay needs the graphs captured"));
        }
        self.path = path;
        Ok(())
    }

    /// The rungs: `(draft width, rows)`.
    pub fn rungs(&self) -> Vec<(u32, u32)> {
        self.rungs.iter().map(|r| (r.width, r.rows)).collect()
    }

    /// Lays out every rung and, with graphs on, captures each: run directly over an
    /// all-padding step, captured, replayed once over it.
    fn prepare_rungs(&mut self, capture: bool) -> Result<()> {
        let k = self.cfg.draft_tokens;
        let mut ladders = vec![(0, crate::mimo::draft::draft_ladder(&self.cfg.buckets, 0))];
        if k > 0 {
            ladders.push((k, crate::mimo::draft::draft_ladder(&self.cfg.buckets, k)));
        }
        let mut rungs = Vec::new();
        for (w, ladder) in ladders {
            for rows in ladder {
                if rows as usize > self.scratch.caps.drafting.max(1) {
                    continue;
                }
                let shape = self.rung_shape(rows, w)?;
                let layout = Layout::new(&shape);
                rungs.push(Rung {
                    width: w,
                    rows,
                    shape,
                    layout,
                    graph: None,
                });
            }
        }
        let words = rungs.iter().map(|r| r.layout.words).max().unwrap_or(64);
        self.scratch.table = Alloc::new(&self.dev, words * 4)?;
        self.scratch.caps.table = words;
        for r in &rungs {
            self.check_rung_fits(&r.shape)?;
        }
        self.rungs = rungs;
        let geo = self.geo();
        for g in 0..self.rungs.len() {
            let (w, rows) = (self.rungs[g].width, self.rungs[g].rows);
            let pads: Vec<Row> = (0..rows).map(|_| self.pad_row(w)).collect();
            let (p, _) = self.plan(&pads, k > 0);
            let layout = self.rungs[g].layout.clone();
            // Every rung's all-padding step packs into its layout, on any device.
            let words = plan::pack(&p, &layout, &geo)?;
            if capture {
                self.upload_table(&words, true)?;
                self.program(&layout)?;
                self.check_pad_status(&layout, w, rows)?;
                let this = &*self;
                let graph = this.dev.capture(&mut || this.program(&layout))?;
                self.dev.replay(graph.as_ref())?;
                self.check_pad_status(&layout, w, rows)?;
                self.rungs[g].graph = Some(graph);
            }
        }
        if capture {
            self.path = DecodePath::Replay;
        }
        Ok(())
    }

    fn check_pad_status(&self, l: &Layout, w: u32, rows: u32) -> Result<()> {
        let rb = self.readback(l.readback_words())?;
        if rb[0] != 0 {
            return Err(CudaError::new(format!(
                "rung of {rows} rows at width {w}: status {:#x} over padding",
                rb[0]
            )));
        }
        Ok(())
    }

    /// A rung's every list within the scratch's pad rows and capacities.
    fn check_rung_fits(&self, s: &Shape) -> Result<()> {
        let c = &self.scratch.caps;
        let bad = |what: &str| Err(CudaError::new(format!("rung past the scratch: {what}")));
        let t = &s.target;
        if t.staged > c.staged || t.conv_rows > c.conv_rows || t.attn.rows > c.rows {
            return bad("the target's rows");
        }
        if t.tiles > c.tiles {
            return bad("the blocked DeltaNet's blocks");
        }
        if t.stage_conv.max(t.post_conv) > c.pad_conv || t.stage_ab.max(t.post_ab) > c.pad_ab {
            return bad("pad rows");
        }
        if s.keep.max(s.head) > c.pad_h || s.head > c.head {
            return bad("kept or head rows");
        }
        Ok(())
    }

    fn save_state(&mut self, slot: Slot, len: u32, buffer: u32) -> Result<()> {
        assert!(
            self.slots[slot as usize].base.is_some(),
            "a state saved from a slot that holds none"
        );
        let (base, _) = self.slots[slot as usize].base.expect("checked");
        assert!(
            len.is_multiple_of(self.cfg.block_size),
            "a state saved off a boundary"
        );
        assert!(
            buffer == base
                || (self.state_tags[buffer as usize] == 0
                    && !self.state_encrypted[buffer as usize]),
            "state buffer {buffer}: saved into while it holds a state"
        );
        let (src, src_pos) = self.source(slot, len, len + 1);
        let run = Run {
            slot,
            start: len,
            host: Vec::new(),
            lanes: Vec::new(),
            src,
            src_pos,
            writes: vec![(len, buffer)],
            live_store: false,
            tail_from: u32::MAX,
        };
        let p = StepPlan {
            target: self.target_pass(std::slice::from_ref(&run), false),
            ..StepPlan::default()
        };
        let geo = self.geo();
        let layout = Layout::new(&Shape::of(&p, &geo)?);
        let words = plan::pack(&p, &layout, &geo)?;
        self.upload_table(&words, false)?;
        self.fold_program(&layout.target)?;
        self.dev.synchronize()?;
        self.set_tag(buffer, len)
    }
}

impl<D: QwenDevice> Executor for QwenCudaExecutor<D> {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn execute(&mut self, step: &StepInput) -> std::result::Result<StepOutput, ExecutorError> {
        self.steps += 1;
        if n32(step.seqs.len()) > step.bucket.max_seqs
            || step.query_tokens() > step.bucket.max_tokens
        {
            return Err(ExecutorError("batch exceeds its bucket".into()));
        }
        self.step(step).map_err(ex)
    }
}

// ---- what the tests read back ----

impl<D: QwenDevice> QwenCudaExecutor<D> {
    fn read_words(&self, addr: u64, bytes: usize) -> Vec<u32> {
        let mut out = vec![0u8; bytes];
        // SAFETY: a range the caller took from a live allocation.
        unsafe { self.dev.download(addr, &mut out) }.expect("download");
        out.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// A state buffer's image as words, in its cipher's order: the data, then the tag
    /// (the reference executor's `block_image` of the state pool, word for word).
    pub fn state_image(&self, b: u32) -> Vec<u32> {
        let mut w = self.read_words(self.state_addr(b), self.data_bytes());
        w.extend(self.read_words(self.state_addr(b) + self.data_bytes() as u64, 4));
        w
    }

    /// Whether a block (or a state buffer, at the state pool's index) is all zero bytes
    /// and not encrypted.
    pub fn block_is_zero(&self, group: u32, b: u32) -> bool {
        !self.block_is_encrypted(group, b)
            && self
                .block_ranges(group, b)
                .into_iter()
                .all(|(a, n)| self.read_words(a, n).iter().all(|&x| x == 0))
    }

    pub fn block_is_encrypted(&self, group: u32, b: u32) -> bool {
        if group == self.state_pool() {
            self.state_encrypted[b as usize]
        } else {
            self.kv_encrypted[group as usize][b as usize]
        }
    }

    pub fn host_block_is_zero(&self, group: u32, host: u32) -> bool {
        let enc = if group == self.state_pool() {
            self.host_state_encrypted[host as usize]
        } else {
            self.host_kv_encrypted[group as usize][host as usize]
        };
        let ptr = self.host_block_ptr(group, host);
        // SAFETY: the host block's bytes; nothing is queued against them between steps.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, self.host_block_bytes(group)) };
        !enc && bytes.iter().all(|&b| b == 0)
    }

    pub fn host_block_is_encrypted(&self, group: u32, host: u32) -> bool {
        if group == self.state_pool() {
            self.host_state_encrypted[host as usize]
        } else {
            self.host_kv_encrypted[group as usize][host as usize]
        }
    }

    /// Whether a slot holds no per-sequence state: no block mapped, no base, and its live
    /// image, tail and kept states all zero.
    pub fn slot_state_is_zero(&self, slot: Slot) -> bool {
        let s = slot as usize;
        let k1 = self.cfg.draft_tokens as usize + 1;
        self.tables[s].iter().flatten().all(|&b| b == NULL_BLOCK)
            && self.slots[s].base.is_none()
            && self
                .read_words(self.live_addr(slot), self.image_stride)
                .iter()
                .all(|&x| x == 0)
            && self
                .tail_ranges(slot)
                .into_iter()
                .all(|(a, n)| self.read_words(a, n).iter().all(|&x| x == 0))
            && self
                .read_words(self.hidden_row(slot, 0), k1 * self.dims.hidden as usize * 2)
                .iter()
                .all(|&x| x == 0)
    }

    /// The state pool's index in maintenance.
    pub fn state_pool_index(&self) -> u32 {
        self.state_pool()
    }
}

// ---- a measurement aid: one sequence's logits at every position ----

impl<D: QwenDevice> QwenCudaExecutor<D> {
    /// The target's logits at every position of `tokens`, computed as one prefill from the
    /// zero state in slot `slot` over KV blocks `1 ..`, which must be unused: what the
    /// real-weights measurements compare with the model crate's forward. The slot and the
    /// blocks are zeroed afterwards. The serving path never calls it.
    pub fn dense_logits(&mut self, slot: Slot, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let n = u32of(tokens.len(), "tokens")?;
        let bs = self.cfg.block_size;
        let blocks = n.div_ceil(bs);
        if blocks + 1 > self.cfg.num_blocks || n > self.cfg.max_model_len {
            return Err(CudaError::new("the sequence does not fit the pools"));
        }
        assert!(
            self.slots[slot as usize].base.is_none(),
            "slot {slot} is in use"
        );
        let c = self.scratch.caps;
        // Pieces of at most one head launch's rows: the logits are the vocabulary a row.
        // In blocks, a whole number of blocks a piece, the live state between them a
        // boundary's: a scratch of fewer rows than a block cannot take one.
        let mut piece = c.rows.min(c.head).min(c.logits_t).max(1);
        if self.blocked() {
            let ch = self.chunk() as usize;
            piece = piece / ch * ch;
            if piece == 0 {
                return Err(CudaError::new(
                    "dense logits in blocks need the scratch's rows for a whole block",
                ));
            }
        }
        for (i, b) in (1..=blocks).enumerate() {
            self.tables[slot as usize][TARGET][i] = b;
        }
        self.slots[slot as usize].base = Some((NULL_STATE, 0));
        let vocab = self.dims.vocab as usize;
        let geo = self.geo();
        let mut logits = Vec::with_capacity(tokens.len());
        for (i, chunk) in tokens.chunks(piece).enumerate() {
            let start = n32(i * piece);
            let m = n32(chunk.len());
            let run = Run {
                slot,
                start,
                host: chunk.to_vec(),
                lanes: Vec::new(),
                src: if start == 0 {
                    Src::Zero
                } else {
                    Src::Live(slot)
                },
                src_pos: start,
                writes: Vec::new(),
                live_store: !self.blocked() || (start + m).is_multiple_of(self.chunk()),
                tail_from: u32::MAX,
            };
            let p = StepPlan {
                target: self.target_pass(std::slice::from_ref(&run), true),
                head: (0..m)
                    .map(|j| Item::new(buf::H_NORMED, j, buf::H_DST, j))
                    .collect(),
                ..StepPlan::default()
            };
            let layout = Layout::new(&Shape::of(&p, &geo)?);
            let words = plan::pack(&p, &layout, &geo)?;
            self.upload_table(&words, false)?;
            self.program(&layout)?;
            let mut host = vec![0f32; chunk.len() * vocab];
            // SAFETY: `m · vocab` f32 the head wrote.
            unsafe {
                let dst =
                    std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), host.len() * 4);
                self.dev.download(self.scratch.logits_t.addr, dst)?;
            }
            logits.extend(host.chunks(vocab).map(<[f32]>::to_vec));
        }
        for b in 1..=blocks {
            self.zero_block(0, b)?;
        }
        self.reset_slot(slot)?;
        self.dev.synchronize()?;
        Ok(logits)
    }
}

/// The bytes of step scratch an executor of `cfg` over a model of `config` allocates at
/// construction in `profile` (the DeltaNet in blocks in the fast profile), its step tables
/// aside: what the node's deployment reserve is held to.
pub fn step_scratch_bytes(
    config: &QwenConfig,
    cfg: &QwenCudaExecutorConfig,
    profile: QwenProfile,
) -> Result<u64> {
    let dims = Dims::of(config)?;
    let tail_cap = cfg.block_size - 1 + cfg.draft_tokens;
    let fast = profile == QwenProfile::Fast;
    program::scratch_bytes(&dims, dims.k_max(), capacities(cfg, tail_cap), fast, fast)
}

/// The blocked DeltaNet's block length for KV blocks of `block_size` positions: its
/// largest power-of-two divisor, at most 16.
pub(super) fn block_chunk(block_size: u32) -> u32 {
    (block_size & block_size.wrapping_neg()).clamp(1, CHUNK_TILE)
}

/// Blocks of the blocked DeltaNet a pass of `staged` positions over `runs` runs can hold:
/// each run's are `⌈count / chunk⌉`.
pub(super) fn chunk_tiles(staged: usize, runs: usize, chunk: u32) -> usize {
    staged / chunk as usize + runs
}

fn lcm(a: usize, b: usize) -> usize {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
}

/// What the scratch is sized for: the largest bucket's rows and tokens (drafts among
/// them), every slot's drafts, and any decode rung's padded lists.
fn capacities(cfg: &QwenCudaExecutorConfig, tail_cap: u32) -> Caps {
    let r = cfg.buckets.iter().map(|b| b.max_seqs).max().unwrap_or(1) as usize;
    let t = cfg.buckets.iter().map(|b| b.max_tokens).max().unwrap_or(1) as usize;
    let k = cfg.draft_tokens as usize;
    let tc = tail_cap as usize;
    let bs = cfg.block_size as usize;
    // The target's rows: a step's host tokens and drafts, which its bucket bounds.
    let rows = t.max(r * (k + 1));
    let staged = (t + r * tc).max(r * (k + 1 + tc));
    let positions = cfg.max_model_len + cfg.draft_tokens + 1;
    Caps {
        rows,
        staged,
        conv_rows: 3 * r + staged,
        drafter: rows,
        head: (r * (k + 1)).max(64),
        logits_t: (r * (k + 1)).max(64),
        logits_d: r,
        drafting: r,
        depth: k,
        plain: r,
        targets: r * (k + 1),
        pad_conv: 3 * r + staged + 3 * SEQ_WRITES * r + 3 * r + r * (k + 1),
        pad_ab: staged + r * (k + 1),
        pad_o: r.max(r * k),
        pad_h: r * (k + 1) + r * ((k + 1) / bs.max(1) + 1) + r * k + r,
        pad_t: r * k + r,
        table: 64,
        tiles: chunk_tiles(staged, r + 1, block_chunk(cfg.block_size)),
        readback: 1 + r + r * (k + 1) + r,
        score_stride: positions as usize,
        // A rung's rows: every sequence's host token and drafts.
        partial_rows: plan::partial_rows(n32(rows), n32(r * (k + 1)), positions) as usize,
    }
}
