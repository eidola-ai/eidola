//! [`QwenCudaExecutor`]: the serving core's [`Executor`] for Qwen3.8 on a [`QwenDevice`].
//!
//! It reproduces `eidola-engine-cpu`'s `QwenCpuExecutor` (the contract every device
//! executor is held to), step for step: the same maintenance, the same rows, the same
//! state images byte for byte, and in the exact profile the same logits bit for bit. What
//! differs is only where the numbers are computed (the kernel families of
//! `csrc/qwen_abi.cuh`, batched over a step's rows) and how the recurrent state between
//! steps is kept ([`StateStrategy`]).
//!
//! A step:
//!
//! 1. Maintenance in order, then table updates.
//! 2. **Pass A**: the target over every row's host tokens, all rows in one forward. A
//!    DeltaNet layer's recurrence starts each row from its source (the slot's live state,
//!    or its base buffer), folds the kept positions between the source and the row's
//!    `context_len` without output, runs the host tokens with output, stores every named
//!    state write as the recurrence passes it, and stores the state after the host tokens
//!    into the slot's live image.
//! 3. The drafter's rows over the host positions (with drafting), then the draft chain,
//!    depth by depth over every drafting row.
//! 4. **Pass B**: the target over the drafts, from the live image; nothing stored but the
//!    drafts' KV in reserved positions and their inputs in the tail.
//! 5. Sampling and chain acceptance on the host (`eidola_engine::sampling`, as the
//!    reference executor).
//! 6. The drafter's canonical rewrite at the accepted drafts, the boundary taps, the kept
//!    normed states and the tail.

use std::any::Any;
use std::collections::HashMap;

use eidola_engine::executor::{
    Executor, ExecutorError, Maintenance, SeqEntry, Slot, StateRow, StateWrite, StepInput,
    StepOutput,
};
use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::sampling::{self, Logits, SamplingParams, Stream};
use eidola_engine::secret::KvKey;
use eidola_engine::spec::{
    AttentionKind, Bucket, Coverage, KvGroupSpec, KvRole, ModelSpec, NULL_BLOCK, NULL_STATE,
    StateGroupSpec, StateSpec,
};
use eidola_engine_model::qwen::QwenModel;
use eidola_engine_model::qwen::config::LayerKind;

use super::abi::*;
use super::device::{QwenDevice, QwenProfile};
use super::weights::{DevWeights, Fp8W, LayerW, MixerW};
use crate::{CudaError, Result, narrow};

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
    /// f32 values one tail position takes per layer: `mixed`, `a`, `b`.
    fn tail_floats(&self) -> usize {
        (self.conv_dim() + 2 * self.v_heads) as usize
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
    /// Live images, one per slot.
    live: Alloc,
    /// Tails: per slot, `[cap][dn layers][tail floats]` f32.
    tails: Alloc,
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
    status: Alloc,
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
        if !c.hidden_size.is_multiple_of(128) || !c.intermediate_size.is_multiple_of(128) {
            return Err(CudaError::new("widths must be whole 128-blocks"));
        }
        let dims = Dims {
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
        };
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
            let data = Alloc::new(&dev, cfg.num_blocks as usize * block_bytes)?;
            let tap_bytes = dims.hidden as usize * 2;
            let taps = if g == DRAFTER {
                Some(Alloc::new(&dev, cfg.num_blocks as usize * tap_bytes)?)
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
        let image_stride = (data_bytes + 4).next_multiple_of(256);
        let states = Alloc::new(&dev, cfg.state_buffers as usize * image_stride)?;
        let host_states = if tiered {
            let (ptr, keep) = dev.alloc_host(cfg.host_state_buffers as usize * image_stride)?;
            Some(HostAlloc { ptr, _keep: keep })
        } else {
            None
        };
        let slots_n = cfg.num_state_slots as usize;
        let live = Alloc::new(&dev, slots_n * image_stride)?;
        let tail_cap = bs - 1 + k;
        let tails = Alloc::new(
            &dev,
            slots_n * tail_cap as usize * dn_layers * dims.tail_floats() * 4,
        )?;
        let hiddens = Alloc::new(&dev, slots_n * (k as usize + 1) * dims.hidden as usize * 2)?;
        let status = Alloc::new(&dev, 8)?;
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
        Ok(Self {
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
            tails,
            tail_cap,
            hiddens,
            spec,
            cfg,
            steps: 0,
            status,
        })
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

    /// A slot's whole tail: its address and bytes (none with one-token blocks and no
    /// drafts).
    fn slot_tail(&self, slot: Slot) -> (u64, usize) {
        let bytes = self.tail_cap as usize * self.dn_layers() * self.dims.tail_floats() * 4;
        (self.tails.addr + u64::from(slot) * bytes as u64, bytes)
    }

    /// The row of tail position `pos` of a DeltaNet layer; only a slot with a tail has one.
    fn tail_row(&self, slot: Slot, pos: u32, layer: usize) -> u64 {
        let tf = self.dims.tail_floats();
        let per_pos = self.dn_layers() * tf;
        let i = u64::from(pos % self.tail_cap) * per_pos as u64 + (layer * tf) as u64;
        self.slot_tail(slot).0 + i * 4
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

    /// XORs a block's image with its keystream: each range from a fresh keystream block,
    /// as `eidola_engine::kv_cipher` defines a block image.
    fn crypt(
        &mut self,
        group: u32,
        b: u32,
        key: &KvKey,
        nonce: &KvNonce,
        encrypt: bool,
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
            // SAFETY: the block's range, whole words.
            unsafe { self.dev.cipher(addr, words, counter, key, nonce)? };
            counter += words.div_ceil(16);
        }
        Ok(())
    }

    fn reset_slot(&mut self, slot: Slot) -> Result<()> {
        let s = slot as usize;
        for g in &mut self.tables[s] {
            g.fill(NULL_BLOCK);
        }
        self.slots[s] = SlotRec::default();
        let (tail, tail_bytes) = self.slot_tail(slot);
        let k1 = self.cfg.draft_tokens as usize + 1;
        // SAFETY: the slot's own ranges of the live images, tails and hiddens.
        unsafe {
            self.dev.zero(self.live_addr(slot), self.image_stride)?;
            self.dev.zero(tail, tail_bytes)?;
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

    fn maintain(&mut self, step: &StepInput) -> Result<()> {
        let state = self.state_pool();
        for m in &step.maintenance {
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
                    self.crypt(group, block, key, &nonce, encrypt)?;
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
        Ok(())
    }

    /// The source a slot's recurrence starts from for positions up to `upto`: its live
    /// image when the strategy keeps one and it stands at or before `upto` past the base,
    /// else its base buffer (the zero state for `NULL_STATE`). `(address, position)`.
    fn source(&self, slot: Slot, upto: u32, writes_after: u32) -> (u64, u32) {
        let rec = &self.slots[slot as usize];
        let (base, base_len) = rec.base.expect("a slot with a base");
        if self.cfg.strategy == StateStrategy::Live
            && let Some(lp) = rec.live_pos
            && lp >= base_len
            && lp <= upto
            && lp < writes_after
        {
            return (self.live_addr(slot), lp);
        }
        if base == NULL_STATE {
            assert_eq!(
                base_len, 0,
                "the zero state named as the state at {base_len}"
            );
            return (0, 0);
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
        (self.state_addr(base), base_len)
    }

    fn save_state(&mut self, slot: Slot, len: u32, buffer: u32) -> Result<()> {
        let (base, _) = self.slots[slot as usize]
            .base
            .expect("a state saved from a slot that holds none");
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
            tokens: Vec::new(),
            src,
            src_pos,
            writes: vec![(len, buffer)],
            live_store: false,
            tail_from: u32::MAX,
            kv: false,
        };
        self.dn_only(&run)?;
        self.set_tag(buffer, len)
    }
}

// ---- the forward ----

/// One sequence's positions through the target (or, with no tokens, a fold alone).
#[derive(Clone, Debug)]
struct Run {
    slot: Slot,
    /// The first new position; the run's tokens are at `start ..`.
    start: u32,
    tokens: Vec<u32>,
    /// The DeltaNet source image (0: the zero state) and the position it stands at; the
    /// kept positions `src_pos .. start` are folded from the tail first.
    src: u64,
    src_pos: u32,
    /// State writes `(len, buffer)`, ascending.
    writes: Vec<(u32, u32)>,
    /// Store the state after the run's tokens into the slot's live image.
    live_store: bool,
    /// New positions at or past this join the tail.
    tail_from: u32,
    /// Write the target's KV at the new positions.
    kv: bool,
}

impl Run {
    fn fold(&self) -> u32 {
        self.start - self.src_pos
    }
    fn n(&self) -> u32 {
        n32(self.tokens.len())
    }
    fn end(&self) -> u32 {
        self.start + self.n()
    }
}

/// A forward's scratch: allocated per forward, zeroed.
struct Scratch {
    bufs: Vec<Alloc>,
}

impl Scratch {
    fn get<D: QwenDevice>(&mut self, dev: &D, bytes: usize) -> Result<u64> {
        let a = Alloc::new(dev, bytes)?;
        let addr = a.addr;
        self.bufs.push(a);
        Ok(addr)
    }
}

/// Where a forward's runs sit.
struct Layout {
    /// Row offset of each run's new positions.
    rows: Vec<u32>,
    /// Staged-position offset of each run (its folds, then its new positions).
    staged: Vec<u32>,
    /// Conv-row offset of each run (three history rows, then the staged ones).
    conv: Vec<u32>,
    t: u32,
    p: u32,
    conv_rows: u32,
}

impl Layout {
    fn of(runs: &[Run]) -> Layout {
        let (mut rows, mut staged, mut conv) = (Vec::new(), Vec::new(), Vec::new());
        let (mut t, mut p, mut c) = (0u32, 0u32, 0u32);
        for r in runs {
            rows.push(t);
            staged.push(p);
            conv.push(c);
            t += r.n();
            p += r.fold() + r.n();
            c += 3 + r.fold() + r.n();
        }
        Layout {
            rows,
            staged,
            conv,
            t,
            p,
            conv_rows: c,
        }
    }
}

impl<D: QwenDevice> QwenCudaExecutor<D> {
    fn norm(&self, x: u64, add: u64, w: u64, out: u64, rows: u32, n: u32) -> Result<()> {
        let p = NormParams {
            x,
            add,
            weight: w,
            out,
            rows,
            units: 1,
            n,
            x_stride: n,
            add_stride: n,
            out_stride: n,
            eps: self.dims.eps,
            ..NormParams::default()
        };
        // SAFETY: `rows · n` BF16 values at `x`, `add` and `out`; `n` f32 at `w`.
        unsafe { self.dev.norm(&p) }
    }

    /// The FP8 linear of `rows` BF16 rows at `x` (stride `w.k`) into `out` (stride `w.n`).
    fn lin8(&self, sc: &mut Scratch, x: u64, rows: u32, w: &Fp8W, out: u64) -> Result<()> {
        let m_pad = rows.next_multiple_of(4).max(4);
        let codes = sc.get(&self.dev, m_pad as usize * w.k as usize)?;
        let scales = sc.get(&self.dev, (w.k / BLOCK) as usize * m_pad as usize * 4)?;
        let q = QuantParams {
            x,
            codes,
            scales,
            rows,
            k: w.k,
            x_stride: w.k,
            m_pad,
        };
        let l = Lin8Params {
            a: codes,
            a_scales: scales,
            w: w.codes,
            w_scales: w.scales,
            out,
            rows,
            n: w.n,
            k: w.k,
            m_pad,
            out_stride: w.n,
            reserved: 0,
        };
        // SAFETY: scratch sized for `m_pad` rows; `out` holds `m_pad · n` (the fast form
        // writes the padding rows).
        unsafe {
            self.dev.quant(&q)?;
            self.dev.lin8(&l)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        x: u64,
        rows: u32,
        w: u64,
        n: u32,
        k: u32,
        out: u64,
        f32_out: bool,
        large: bool,
    ) -> Result<()> {
        let p = LinearParams {
            x,
            w,
            out,
            rows,
            n,
            k,
            x_stride: k,
            out_stride: n,
            out_f32: u32::from(f32_out),
        };
        // SAFETY: the caller's buffers.
        unsafe { self.dev.linear(&p, large) }
    }

    fn widen(
        &self,
        x: u64,
        x_stride: u32,
        out: u64,
        out_stride: u32,
        rows: u32,
        width: u32,
    ) -> Result<()> {
        let p = WidenParams {
            x,
            out,
            rows,
            width,
            x_stride,
            out_stride,
        };
        // SAFETY: the caller's buffers.
        unsafe { self.dev.widen(&p) }
    }

    /// The MLP of `rows` normed rows `x` into `m`.
    fn mlp(&self, sc: &mut Scratch, l: &LayerW, x: u64, rows: u32, m: u64) -> Result<()> {
        let pad = rows.next_multiple_of(4).max(4) as usize;
        let inter = self.dims.inter;
        let g = sc.get(&self.dev, pad * inter as usize * 2)?;
        let u = sc.get(&self.dev, pad * inter as usize * 2)?;
        let hmid = sc.get(&self.dev, pad * inter as usize * 2)?;
        self.lin8(sc, x, rows, &l.gate, g)?;
        self.lin8(sc, x, rows, &l.up, u)?;
        let p = GateParams {
            a: u,
            g,
            out: hmid,
            rows,
            width: inter,
            a_stride: inter,
            g_stride: inter,
            out_stride: inter,
            unit: inter,
            g_unit_stride: 0,
            kind: GATE_SWIGLU,
        };
        // SAFETY: `rows · inter` values in each.
        unsafe { self.dev.gate(&p)? };
        self.lin8(sc, hmid, rows, &l.down, m)
    }
}

// ---- the DeltaNet layer ----

/// A DeltaNet layer's per-forward arrays, shared by its layers.
struct DnBufs {
    conv_in: u64,
    ab: u64,
    q: u64,
    k: u64,
    v: u64,
    beta: u64,
    alpha: u64,
    o: u64,
    window: u64,
    seqs: u64,
}

impl<D: QwenDevice> QwenCudaExecutor<D> {
    fn dn_bufs(&self, sc: &mut Scratch, runs: &[Run], lay: &Layout) -> Result<DnBufs> {
        let d = &self.dims;
        let p = lay.p.max(1) as usize;
        let mut window = Vec::with_capacity(lay.p as usize);
        for (i, r) in runs.iter().enumerate() {
            for j in 0..r.fold() + r.n() {
                window.push(lay.conv[i] + j);
            }
        }
        let bufs = DnBufs {
            conv_in: sc.get(
                &self.dev,
                lay.conv_rows.max(1) as usize * d.conv_dim() as usize * 4,
            )?,
            ab: sc.get(&self.dev, p * 2 * d.v_heads as usize * 4)?,
            q: sc.get(&self.dev, p * d.key_dim() as usize * 4)?,
            k: sc.get(&self.dev, p * d.key_dim() as usize * 4)?,
            v: sc.get(&self.dev, p * d.value_dim() as usize * 4)?,
            beta: sc.get(&self.dev, p * d.v_heads as usize * 4)?,
            alpha: sc.get(&self.dev, p * d.v_heads as usize * 4)?,
            o: sc.get(&self.dev, p * d.value_dim() as usize * 4)?,
            window: sc.get(&self.dev, p * 4)?,
            seqs: sc.get(&self.dev, runs.len().max(1) * std::mem::size_of::<DnSeq>())?,
        };
        let mut seqs = Vec::with_capacity(runs.len());
        for (i, r) in runs.iter().enumerate() {
            let mut s = DnSeq {
                src: r.src,
                live: if r.live_store {
                    self.live_addr(r.slot)
                } else {
                    0
                },
                first: lay.staged[i],
                count: r.fold() + r.n(),
                out_from: r.fold(),
                live_after: r.fold() + r.n(),
                ..DnSeq::default()
            };
            if r.writes.len() > SEQ_WRITES {
                return Err(CudaError::new(
                    "more state writes in one row than the kernel takes",
                ));
            }
            for (j, &(len, buf)) in r.writes.iter().enumerate() {
                s.write_after[j] = len - r.src_pos;
                s.write_dst[j] = self.state_addr(buf);
            }
            s.writes = n32(r.writes.len());
            seqs.push(s);
        }
        // SAFETY: the arrays were sized above.
        unsafe {
            self.dev.upload(bufs.window, bytes_of(&window))?;
            self.dev.upload(bufs.seqs, bytes_of(&seqs))?;
        }
        Ok(bufs)
    }

    /// Stages one DeltaNet layer's inputs: per run its source's conv history, its folded
    /// positions' kept inputs from the tail, and its new positions' `mixed`, `a` and `b`
    /// (`mixed`, `a`, `b` BF16 rows at the run's rows; null for a fold alone).
    #[allow(clippy::too_many_arguments)]
    fn dn_stage(
        &self,
        runs: &[Run],
        lay: &Layout,
        b: &DnBufs,
        dn: usize,
        mixed: u64,
        a: u64,
        bb: u64,
    ) -> Result<()> {
        let d = &self.dims;
        let (cd, vh) = (d.conv_dim() as usize, d.v_heads as usize);
        let hist = 3 * cd * 4;
        for (i, r) in runs.iter().enumerate() {
            let conv0 = b.conv_in + u64::from(lay.conv[i]) * (cd * 4) as u64;
            // SAFETY: rows of the staging arrays and the slot's tail and source image.
            unsafe {
                if r.src == 0 {
                    self.dev.zero(conv0, hist)?;
                } else {
                    self.dev
                        .copy(conv0, r.src + (dn * self.layer_bytes()) as u64, hist)?;
                }
                for j in 0..r.fold() {
                    let t = self.tail_row(r.slot, r.src_pos + j, dn);
                    let row = conv0 + u64::from(3 + j) * (cd * 4) as u64;
                    self.dev.copy(row, t, cd * 4)?;
                    let abr = b.ab + u64::from(lay.staged[i] + j) * (2 * vh * 4) as u64;
                    self.dev.copy(abr, t + (cd * 4) as u64, 2 * vh * 4)?;
                }
            }
            if r.n() > 0 {
                let (cdu, vhu) = (d.conv_dim(), d.v_heads);
                let row0 = u64::from(lay.rows[i]);
                let conv_new = conv0 + u64::from(3 + r.fold()) * (cd * 4) as u64;
                let ab_new = b.ab + u64::from(lay.staged[i] + r.fold()) * (2 * vh * 4) as u64;
                let two = u32of(2 * vh, "ab")?;
                self.widen(
                    mixed + row0 * (cd * 2) as u64,
                    cdu,
                    conv_new,
                    cdu,
                    r.n(),
                    cdu,
                )?;
                self.widen(a + row0 * (vh * 2) as u64, vhu, ab_new, two, r.n(), vhu)?;
                self.widen(
                    bb + row0 * (vh * 2) as u64,
                    vhu,
                    ab_new + (vh * 4) as u64,
                    two,
                    r.n(),
                    vhu,
                )?;
            }
        }
        Ok(())
    }

    /// The recurrence of one DeltaNet layer over the staged positions, its stores, and
    /// what follows it: the conv history of every store and the tail.
    fn dn_recur(
        &self,
        runs: &[Run],
        lay: &Layout,
        b: &DnBufs,
        dn: usize,
        w: &super::weights::DeltaNetW,
    ) -> Result<()> {
        let d = &self.dims;
        let prep = DnPrepParams {
            conv_in: b.conv_in,
            window: b.window,
            ab: b.ab,
            conv_w: w.conv,
            exp_a_log: w.exp_a_log,
            dt_bias: w.dt_bias,
            q: b.q,
            k: b.k,
            v: b.v,
            beta: b.beta,
            alpha: b.alpha,
            positions: lay.p,
            k_heads: d.k_heads,
            v_heads: d.v_heads,
            k_dim: d.k_dim,
            v_dim: d.v_dim,
            reserved: 0,
        };
        let rec = DnRecParams {
            seqs: b.seqs,
            q: b.q,
            k: b.k,
            v: b.v,
            beta: b.beta,
            alpha: b.alpha,
            out: b.o,
            state_offset: (dn * self.layer_bytes() + 3 * d.conv_dim() as usize * 4) as u64,
            seq_count: u32of(runs.len(), "runs")?,
            k_heads: d.k_heads,
            v_heads: d.v_heads,
            k_dim: d.k_dim,
            v_dim: d.v_dim,
            reserved: 0,
        };
        // SAFETY: the staged arrays hold `lay.p` positions; every state address is a live
        // image or a state buffer.
        unsafe {
            self.dev.dn_prep(&prep)?;
            self.dev.dn_rec(&rec)?;
        }
        let (cd, vh) = (d.conv_dim() as usize, d.v_heads as usize);
        for (i, r) in runs.iter().enumerate() {
            let conv0 = b.conv_in + u64::from(lay.conv[i]) * (cd * 4) as u64;
            // The conv history after `len`: positions `len - 3 .. len`, staged rows
            // `3 + (len - 3 - src_pos) ..`.
            let hist_at = |len: u32| conv0 + u64::from(len - r.src_pos) * (cd * 4) as u64;
            let layer_off = (dn * self.layer_bytes()) as u64;
            // SAFETY: staging rows and image ranges.
            unsafe {
                for &(len, buf) in &r.writes {
                    self.dev
                        .copy(self.state_addr(buf) + layer_off, hist_at(len), 3 * cd * 4)?;
                }
                if r.live_store {
                    self.dev.copy(
                        self.live_addr(r.slot) + layer_off,
                        hist_at(r.end()),
                        3 * cd * 4,
                    )?;
                }
                for j in 0..r.n() {
                    let pos = r.start + j;
                    if pos < r.tail_from {
                        continue;
                    }
                    let t = self.tail_row(r.slot, pos, dn);
                    let row = conv0 + u64::from(3 + r.fold() + j) * (cd * 4) as u64;
                    self.dev.copy(t, row, cd * 4)?;
                    let abr = b.ab + u64::from(lay.staged[i] + r.fold() + j) * (2 * vh * 4) as u64;
                    self.dev.copy(t + (cd * 4) as u64, abr, 2 * vh * 4)?;
                }
            }
        }
        Ok(())
    }

    /// A fold alone, every DeltaNet layer (`SaveState`).
    fn dn_only(&self, run: &Run) -> Result<()> {
        let runs = std::slice::from_ref(run);
        let lay = Layout::of(runs);
        let mut sc = Scratch { bufs: Vec::new() };
        let b = self.dn_bufs(&mut sc, runs, &lay)?;
        let mut dn = 0;
        for (l, kind) in self.dims.kinds.iter().enumerate() {
            if *kind != LayerKind::DeltaNet {
                continue;
            }
            let MixerW::DeltaNet(w) = self.weights.layers[l].mixer else {
                unreachable!("a DeltaNet layer's weights")
            };
            self.dn_stage(runs, &lay, &b, dn, 0, 0, 0)?;
            self.dn_recur(runs, &lay, &b, dn, &w)?;
            dn += 1;
        }
        self.dev.synchronize()
    }
}

/// A slice's bytes.
fn bytes_of<T: Copy>(x: &[T]) -> &[u8] {
    // SAFETY: any initialized `Copy` value's bytes may be read; the ABI structs are
    // `repr(C)` with no padding that carries data the kernels read.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}

// ---- attention, the target forward, the drafter and the head ----

/// Per-row attention inputs of one forward: RoPE positions, query positions, KV
/// targets and block tables, uploaded once.
struct AttnRows {
    rope: u64,
    query: u64,
    targets: u64,
    tables: u64,
    table_stride: u32,
    score_stride: u32,
}

/// One drafter row: the token at `s`, the target's (or a chain step's) normed state
/// before it at `hidden` (a BF16 row on the device).
#[derive(Clone, Copy, Debug)]
struct MtpRow {
    slot: Slot,
    s: u32,
    token: u32,
    hidden: u64,
}

impl<D: QwenDevice> QwenCudaExecutor<D> {
    fn pad(rows: u32) -> usize {
        rows.next_multiple_of(4).max(4) as usize
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

    fn attn_rows(
        &self,
        sc: &mut Scratch,
        group: usize,
        rows: &[(Slot, u32, u32)],
        write: &[bool],
    ) -> Result<AttnRows> {
        let bs = self.cfg.block_size;
        let width = self.tables[0][group].len();
        let (mut rope, mut query, mut targets) = (Vec::new(), Vec::new(), Vec::new());
        let mut tables: Vec<i32> = Vec::with_capacity(rows.len() * width);
        let mut most = 0;
        for (i, &(slot, q, r)) in rows.iter().enumerate() {
            rope.push(r);
            query.push(q);
            most = most.max(q + 1);
            targets.push(if write[i] {
                self.writable_block(slot, group, q) * bs + q % bs
            } else {
                NO_TARGET
            });
            for &b in &self.tables[slot as usize][group] {
                tables.push(i32::try_from(b).map_err(|_| CudaError::new("block id"))?);
            }
        }
        let a = AttnRows {
            rope: sc.get(&self.dev, rows.len().max(1) * 4)?,
            query: sc.get(&self.dev, rows.len().max(1) * 4)?,
            targets: sc.get(&self.dev, rows.len().max(1) * 4)?,
            tables: sc.get(&self.dev, tables.len().max(1) * 4)?,
            table_stride: u32of(width, "table")?,
            score_stride: most.max(1),
        };
        // SAFETY: sized above.
        unsafe {
            self.dev.upload(a.rope, bytes_of(&rope))?;
            self.dev.upload(a.query, bytes_of(&query))?;
            self.dev.upload(a.targets, bytes_of(&targets))?;
            self.dev.upload(a.tables, bytes_of(&tables))?;
        }
        Ok(a)
    }

    /// One attention layer's mixer over `rows` normed rows `x` into `mix`.
    #[allow(clippy::too_many_arguments)]
    fn attention_mix(
        &self,
        sc: &mut Scratch,
        w: &super::weights::AttentionW,
        x: u64,
        rows: u32,
        group: usize,
        layer: u32,
        key_first: u32,
        ar: &AttnRows,
        mix: u64,
    ) -> Result<()> {
        let d = &self.dims;
        let pad = Self::pad(rows);
        let (hq, hd, kvw) = (d.q_heads, d.head_dim, d.kv_width());
        let qw = hq * hd;
        let qg = sc.get(&self.dev, pad * 2 * qw as usize * 2)?;
        let kk = sc.get(&self.dev, pad * kvw as usize * 2)?;
        let vv = sc.get(&self.dev, pad * kvw as usize * 2)?;
        let q = sc.get(&self.dev, pad * qw as usize * 2)?;
        let kr = sc.get(&self.dev, pad * kvw as usize * 2)?;
        let attn = sc.get(&self.dev, pad * qw as usize * 2)?;
        let y = sc.get(&self.dev, pad * qw as usize * 2)?;
        let scores = sc.get(
            &self.dev,
            rows as usize * hq as usize * ar.score_stride as usize * 4,
        )?;
        self.lin8(sc, x, rows, &w.q, qg)?;
        self.lin8(sc, x, rows, &w.k, kk)?;
        self.lin8(sc, x, rows, &w.v, vv)?;
        let rot =
            |src: u64, units: u32, stride: u32, unit_stride: u32, out: u64, wt: u64| NormParams {
                x: src,
                weight: wt,
                out,
                rope_cos: self.weights.rope_cos,
                rope_sin: self.weights.rope_sin,
                positions: ar.rope,
                rows,
                units,
                n: hd,
                x_stride: stride,
                x_unit_stride: unit_stride,
                out_stride: units * hd,
                out_unit_stride: hd,
                rotary: d.rotary,
                eps: d.eps,
                ..NormParams::default()
            };
        let pool = &self.pools[group];
        let kvp = KvWriteParams {
            k: kr,
            v: vv,
            pool: pool.data.addr,
            targets: ar.targets,
            rows,
            width: kvw,
            k_stride: kvw,
            v_stride: kvw,
            layer,
            layers: pool.layers,
            block_size: self.cfg.block_size,
            reserved: 0,
        };
        let ap = AttnParams {
            q,
            pool: pool.data.addr,
            tables: ar.tables,
            positions: ar.query,
            out: attn,
            scores,
            rows,
            q_heads: hq,
            kv_heads: d.kv_heads,
            head_dim: hd,
            q_stride: qw,
            q_head_stride: hd,
            out_stride: qw,
            table_stride: ar.table_stride,
            layer,
            layers: pool.layers,
            block_size: self.cfg.block_size,
            score_stride: ar.score_stride,
            key_first,
            scale: d.scale,
        };
        let gp = GateParams {
            a: attn,
            g: qg + u64::from(hd) * 2,
            out: y,
            rows,
            width: qw,
            a_stride: qw,
            g_stride: 2 * qw,
            out_stride: qw,
            unit: hd,
            g_unit_stride: 2 * hd,
            kind: GATE_SIGMOID,
        };
        // SAFETY: every buffer was sized for `rows` (padded) above; the targets are blocks
        // the rows' slots alone map, the tables cover every visible position.
        unsafe {
            self.dev.norm(&rot(qg, hq, 2 * qw, 2 * hd, q, w.q_norm))?;
            self.dev.norm(&rot(kk, d.kv_heads, kvw, hd, kr, w.k_norm))?;
            self.dev.kv_write(&kvp)?;
            self.dev.attention(&ap)?;
            self.dev.gate(&gp)?;
        }
        self.lin8(sc, y, rows, &w.o, mix)
    }
}

/// What a target forward leaves: the normed final rows (BF16 `[t, hidden]`) and where each
/// run's rows are; the scratch stays alive with it.
struct Forward {
    normed: u64,
    rows: Vec<u32>,
    _scratch: Scratch,
}

impl<D: QwenDevice> QwenCudaExecutor<D> {
    /// The target over every run's tokens, all runs in one forward.
    fn target(&self, runs: &[Run]) -> Result<Forward> {
        let d = self.dims.clone();
        let lay = Layout::of(runs);
        let mut sc = Scratch { bufs: Vec::new() };
        let t = lay.t;
        let pad = Self::pad(t);
        let hsz = d.hidden as usize;
        let h = sc.get(&self.dev, pad * hsz * 2)?;
        let x = sc.get(&self.dev, pad * hsz * 2)?;
        let mix = sc.get(&self.dev, pad * hsz * 2)?;
        let m = sc.get(&self.dev, pad * hsz * 2)?;
        let normed = sc.get(&self.dev, pad * hsz * 2)?;
        let tokens: Vec<u32> = runs.iter().flat_map(|r| r.tokens.iter().copied()).collect();
        let tok = sc.get(&self.dev, tokens.len().max(1) * 4)?;
        // SAFETY: sized above.
        unsafe { self.dev.upload(tok, bytes_of(&tokens))? };
        self.embed(tok, t, h)?;
        let mut rows = Vec::new();
        let mut write = Vec::new();
        for r in runs {
            for j in 0..r.n() {
                let pos = r.start + j;
                rows.push((r.slot, pos, pos));
                write.push(r.kv);
            }
        }
        let ar = self.attn_rows(&mut sc, TARGET, &rows, &write)?;
        let dnb = self.dn_bufs(&mut sc, runs, &lay)?;
        let (cd, vd, vh) = (
            d.conv_dim() as usize,
            d.value_dim() as usize,
            d.v_heads as usize,
        );
        let mixed = sc.get(&self.dev, pad * cd * 2)?;
        let z = sc.get(&self.dev, pad * vd * 2)?;
        let aa = sc.get(&self.dev, pad * vh * 2)?;
        let bb = sc.get(&self.dev, pad * vh * 2)?;
        let nrm = sc.get(&self.dev, pad * vd * 2)?;
        let (mut dn, mut at) = (0usize, 0u32);
        let mut prev_m = 0u64;
        for (l, lw) in self.weights.layers.iter().enumerate() {
            self.norm(h, prev_m, lw.input_norm, x, t, d.hidden)?;
            match &lw.mixer {
                MixerW::DeltaNet(w) => {
                    self.lin8(&mut sc, x, t, &w.qkv, mixed)?;
                    self.lin8(&mut sc, x, t, &w.z, z)?;
                    self.linear(x, t, w.a, d.v_heads, d.hidden, aa, false, false)?;
                    self.linear(x, t, w.b, d.v_heads, d.hidden, bb, false, false)?;
                    self.dn_stage(runs, &lay, &dnb, dn, mixed, aa, bb)?;
                    self.dn_recur(runs, &lay, &dnb, dn, w)?;
                    for (i, r) in runs.iter().enumerate() {
                        if r.n() == 0 {
                            continue;
                        }
                        let o = dnb.o + u64::from(lay.staged[i] + r.fold()) * (vd * 4) as u64;
                        let p = DnNormParams {
                            o,
                            z: z + u64::from(lay.rows[i]) * (vd * 2) as u64,
                            weight: w.norm,
                            out: nrm + u64::from(lay.rows[i]) * (vd * 2) as u64,
                            rows: r.n(),
                            v_heads: d.v_heads,
                            v_dim: d.v_dim,
                            z_stride: d.value_dim(),
                            eps: d.eps,
                            reserved: 0,
                        };
                        // SAFETY: the run's rows of each array.
                        unsafe { self.dev.dn_norm(&p)? };
                    }
                    self.lin8(&mut sc, nrm, t, &w.out, mix)?;
                    dn += 1;
                }
                MixerW::Attention(w) => {
                    self.attention_mix(&mut sc, w, x, t, TARGET, at, 0, &ar, mix)?;
                    at += 1;
                }
            }
            debug_assert_eq!(
                self.dims.kinds[l] == LayerKind::DeltaNet,
                matches!(lw.mixer, MixerW::DeltaNet(_))
            );
            self.norm(h, mix, lw.post_norm, x, t, d.hidden)?;
            self.mlp(&mut sc, lw, x, t, m)?;
            prev_m = m;
        }
        self.norm(h, prev_m, self.weights.final_norm, normed, t, d.hidden)?;
        Ok(Forward {
            normed,
            rows: lay.rows,
            _scratch: sc,
        })
    }

    fn embed(&self, tokens: u64, rows: u32, out: u64) -> Result<()> {
        let p = EmbedParams {
            tokens,
            table: self.weights.embed,
            out,
            status: self.status.addr,
            rows,
            width: self.dims.hidden,
            vocab: self.dims.vocab,
            out_stride: self.dims.hidden,
        };
        // SAFETY: `rows` tokens and output rows.
        unsafe { self.dev.embed(&p) }
    }

    /// The drafter over `rows`: KV written at each row's position, the normed output
    /// (BF16 `[n, hidden]`) returned with its scratch.
    fn drafter(&self, rows: &[MtpRow]) -> Result<Forward> {
        let mtp = self.weights.mtp.as_ref().expect("a drafter");
        let d = self.dims.clone();
        let n = u32of(rows.len(), "drafter rows")?;
        let mut sc = Scratch { bufs: Vec::new() };
        let pad = Self::pad(n);
        let hsz = d.hidden as usize;
        let tokens: Vec<u32> = rows.iter().map(|r| r.token).collect();
        let tok = sc.get(&self.dev, rows.len().max(1) * 4)?;
        let e = sc.get(&self.dev, pad * hsz * 2)?;
        let hid = sc.get(&self.dev, pad * hsz * 2)?;
        let cat = sc.get(&self.dev, pad * 2 * hsz * 2)?;
        let h = sc.get(&self.dev, pad * hsz * 2)?;
        let x = sc.get(&self.dev, pad * hsz * 2)?;
        let mix = sc.get(&self.dev, pad * hsz * 2)?;
        let m = sc.get(&self.dev, pad * hsz * 2)?;
        let normed = sc.get(&self.dev, pad * hsz * 2)?;
        // SAFETY: sized above; every hidden is a live BF16 row.
        unsafe {
            self.dev.upload(tok, bytes_of(&tokens))?;
            for (i, r) in rows.iter().enumerate() {
                self.dev
                    .copy(hid + (i * hsz * 2) as u64, r.hidden, hsz * 2)?;
            }
        }
        self.embed(tok, n, e)?;
        let half = |src: u64, w: u64, off: u64| NormParams {
            x: src,
            weight: w,
            out: cat + off,
            rows: n,
            units: 1,
            n: d.hidden,
            x_stride: d.hidden,
            out_stride: 2 * d.hidden,
            eps: d.eps,
            ..NormParams::default()
        };
        // SAFETY: `n` rows of each.
        unsafe {
            self.dev.norm(&half(e, mtp.pre_fc_norm_embedding, 0))?;
            self.dev
                .norm(&half(hid, mtp.pre_fc_norm_hidden, (hsz * 2) as u64))?;
        }
        self.linear(cat, n, mtp.fc, d.hidden, 2 * d.hidden, h, false, true)?;
        let ar_rows: Vec<(Slot, u32, u32)> = rows.iter().map(|r| (r.slot, r.s, r.s - 1)).collect();
        let ar = self.attn_rows(&mut sc, DRAFTER, &ar_rows, &vec![true; rows.len()])?;
        let lw = &mtp.layer;
        let MixerW::Attention(w) = &lw.mixer else {
            unreachable!("the drafter's layer is an attention layer")
        };
        self.norm(h, 0, lw.input_norm, x, n, d.hidden)?;
        self.attention_mix(&mut sc, w, x, n, DRAFTER, 0, 1, &ar, mix)?;
        self.norm(h, mix, lw.post_norm, x, n, d.hidden)?;
        self.mlp(&mut sc, lw, x, n, m)?;
        self.norm(h, m, mtp.norm, normed, n, d.hidden)?;
        Ok(Forward {
            normed,
            rows: (0..n).collect(),
            _scratch: sc,
        })
    }

    /// Logits (f32 of BF16 values, the whole head) of the BF16 rows at `rows`.
    fn head(&self, rows: &[u64]) -> Result<Vec<Vec<f32>>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let d = &self.dims;
        let n = u32of(rows.len(), "head rows")?;
        let hsz = d.hidden as usize;
        let mut sc = Scratch { bufs: Vec::new() };
        let x = sc.get(&self.dev, Self::pad(n) * hsz * 2)?;
        let out = sc.get(&self.dev, Self::pad(n) * d.vocab as usize * 4)?;
        // SAFETY: live BF16 rows into the gathered input.
        unsafe {
            for (i, &r) in rows.iter().enumerate() {
                self.dev.copy(x + (i * hsz * 2) as u64, r, hsz * 2)?;
            }
        }
        self.linear(x, n, self.weights.head, d.vocab, d.hidden, out, true, true)?;
        let mut host = vec![0f32; rows.len() * d.vocab as usize];
        // SAFETY: `n · vocab` f32.
        unsafe {
            let dst =
                std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), host.len() * 4);
            self.dev.download(out, dst)?;
        }
        Ok(host.chunks(d.vocab as usize).map(<[f32]>::to_vec).collect())
    }
}

// ---- a step ----

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
    chain_logits: Vec<Vec<f32>>,
    produced: Vec<u32>,
    target_logits: HashMap<u32, Vec<f32>>,
}

impl Row {
    fn e(&self) -> u32 {
        self.p + 1
    }
}

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
            chain_logits: Vec::new(),
            produced: Vec::new(),
            target_logits: HashMap::new(),
        }
    }

    /// Checks a row's base and writes as the reference executor does, and starts a fresh
    /// slot at its base.
    fn check_row(&mut self, row: &Row) {
        let slot = row.slot;
        let bs = self.cfg.block_size;
        let StateRow { base, base_len } = row.base;
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
        for (i, w) in row.writes.iter().enumerate() {
            assert!(
                w.len > base_len && w.len <= row.e() && w.len.is_multiple_of(bs),
                "slot {slot}: a state write at {} outside ({base_len}, {}]",
                w.len,
                row.e()
            );
            assert!(
                i == 0 || row.writes[i - 1].len < w.len,
                "state writes out of order"
            );
            assert!(
                w.buffer != NULL_STATE && w.buffer < self.cfg.state_buffers,
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
                    self.state_tags[w.buffer as usize] == 0
                        && !self.state_encrypted[w.buffer as usize],
                    "state buffer {}: written while it holds a state",
                    w.buffer
                );
            }
        }
    }

    /// The target's normed state at `pos` for `slot`, as a device row: what its last row
    /// kept, or the tap of the drafter block ending there (after a resume).
    fn hidden_before(&self, slot: Slot, pos: u32) -> u64 {
        if let Some(i) = self.slots[slot as usize]
            .hiddens
            .iter()
            .position(|&q| q == pos)
        {
            return self.hidden_row(slot, i);
        }
        let b = self.block_of(slot, DRAFTER, pos);
        let pool = &self.pools[DRAFTER];
        pool.taps.as_ref().expect("taps").addr + u64::from(b) * pool.tap_bytes as u64
    }

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
        for row in &rows {
            self.check_row(row);
        }
        let hsz = u64::from(self.dims.hidden) * 2;
        let hsz_us = self.dims.hidden as usize * 2;
        let drafting = self.pools.len() > DRAFTER;

        // Pass A: every row's host tokens.
        let mut new_base = Vec::with_capacity(rows.len());
        let runs_a: Vec<Run> = rows
            .iter()
            .map(|r| {
                let first_write = r.writes.first().map_or(u32::MAX, |w| w.len);
                let (src, src_pos) = self.source(r.slot, r.c, first_write);
                let (nb, nl) = r
                    .writes
                    .last()
                    .map_or((r.base.base, r.base.base_len), |w| (w.buffer, w.len));
                new_base.push((nb, nl));
                Run {
                    slot: r.slot,
                    start: r.c,
                    tokens: r.toks.clone(),
                    src,
                    src_pos,
                    writes: r.writes.iter().map(|w| (w.len, w.buffer)).collect(),
                    live_store: true,
                    tail_from: nl,
                    kv: true,
                }
            })
            .collect();
        let fa = self.target(&runs_a)?;
        for r in &rows {
            for w in &r.writes {
                self.set_tag(w.buffer, w.len)?;
            }
        }
        let normed_a =
            |i: usize, q: u32, r: &Row| fa.normed + u64::from(fa.rows[i] + q - r.c) * hsz;
        let sampled: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].sample).collect();
        let la = self.head(
            &sampled
                .iter()
                .map(|&i| normed_a(i, rows[i].p, &rows[i]))
                .collect::<Vec<_>>(),
        )?;
        for (&i, l) in sampled.iter().zip(la) {
            let p = rows[i].p;
            rows[i].target_logits.insert(p, l);
        }

        // The drafter over the host positions, then the chain.
        let mut keep_alive = Vec::new();
        if drafting {
            let mut mrows = Vec::new();
            let mut at_p = Vec::new();
            for (i, r) in rows.iter().enumerate() {
                for s in r.c.max(1)..=r.p {
                    let hidden = if s == r.c {
                        self.hidden_before(r.slot, s - 1)
                    } else {
                        normed_a(i, s - 1, r)
                    };
                    if s == r.p && r.k > 0 {
                        at_p.push((i, mrows.len()));
                    }
                    mrows.push(MtpRow {
                        slot: r.slot,
                        s,
                        token: r.toks[(s - r.c) as usize],
                        hidden,
                    });
                }
            }
            if !mrows.is_empty() {
                let fd = self.drafter(&mrows)?;
                let logits = self.head(
                    &at_p
                        .iter()
                        .map(|&(_, m)| fd.normed + m as u64 * hsz)
                        .collect::<Vec<_>>(),
                )?;
                let mut prev: HashMap<usize, u64> = HashMap::new();
                for (&(i, m), l) in at_p.iter().zip(logits) {
                    rows[i].chain_logits.push(l);
                    prev.insert(i, fd.normed + m as u64 * hsz);
                }
                keep_alive.push(fd);
                let depth = rows.iter().map(|r| r.k).max().unwrap_or(0);
                for j in 0..depth {
                    let mut next = Vec::new();
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..rows.len() {
                        if rows[i].k > j {
                            let l = rows[i].chain_logits[j as usize].clone();
                            self.draw(&mut rows[i], &l);
                            if j + 1 < rows[i].k {
                                next.push(i);
                            }
                        }
                    }
                    if next.is_empty() {
                        continue;
                    }
                    let crow: Vec<MtpRow> = next
                        .iter()
                        .map(|&i| MtpRow {
                            slot: rows[i].slot,
                            s: rows[i].p + j + 1,
                            token: rows[i].drafts[j as usize],
                            hidden: prev[&i],
                        })
                        .collect();
                    let fc = self.drafter(&crow)?;
                    let logits = self.head(
                        &(0..next.len())
                            .map(|m| fc.normed + m as u64 * hsz)
                            .collect::<Vec<_>>(),
                    )?;
                    for (m, (&i, l)) in next.iter().zip(logits).enumerate() {
                        rows[i].chain_logits.push(l);
                        prev.insert(i, fc.normed + m as u64 * hsz);
                    }
                    keep_alive.push(fc);
                }
            }
        }

        // Pass B: the drafts, from the live images.
        let with_drafts: Vec<usize> = (0..rows.len())
            .filter(|&i| !rows[i].drafts.is_empty())
            .collect();
        let runs_b: Vec<Run> = with_drafts
            .iter()
            .map(|&i| {
                let r = &rows[i];
                Run {
                    slot: r.slot,
                    start: r.e(),
                    tokens: r.drafts.clone(),
                    src: self.live_addr(r.slot),
                    src_pos: r.e(),
                    writes: Vec::new(),
                    live_store: false,
                    tail_from: new_base[i].1,
                    kv: true,
                }
            })
            .collect();
        let fb = if runs_b.is_empty() {
            None
        } else {
            Some(self.target(&runs_b)?)
        };
        let normed_at = |i: usize, q: u32, rows: &[Row]| -> u64 {
            let r = &rows[i];
            if q <= r.p {
                normed_a(i, q, r)
            } else {
                let b = with_drafts
                    .iter()
                    .position(|&x| x == i)
                    .expect("a drafting row");
                let fb = fb.as_ref().expect("pass B");
                fb.normed + u64::from(fb.rows[b] + q - r.e()) * hsz
            }
        };
        let mut want = Vec::new();
        for &i in &with_drafts {
            for q in rows[i].e()..rows[i].e() + n32(rows[i].drafts.len()) {
                want.push((i, q));
            }
        }
        let lb = self.head(
            &want
                .iter()
                .map(|&(i, q)| normed_at(i, q, &rows))
                .collect::<Vec<_>>(),
        )?;
        for (&(i, q), l) in want.iter().zip(lb) {
            rows[i].target_logits.insert(q, l);
        }

        // Sampling and chain acceptance.
        for r in rows.iter_mut().filter(|r| r.sample) {
            let params = r.params;
            r.produced = if r.k == 0 {
                vec![sampling::sample(
                    self.spec.logits(&r.target_logits[&r.p]),
                    &params,
                    r.p as u64 + 1,
                )]
            } else {
                let p_rows: Vec<Vec<f64>> = (r.p..=r.p + r.k)
                    .map(|pos| {
                        sampling::processed_probs(self.spec.logits(&r.target_logits[&pos]), &params)
                    })
                    .collect();
                sampling::chain_accept(&p_rows, &r.q_rows, &r.drafts, &params, r.p as u64 + 1)
            };
        }

        // The drafter's canonical rows at the accepted drafts, taps and kept states.
        if drafting {
            let mut rw = Vec::new();
            for (i, r) in rows.iter().enumerate() {
                let a = n32(r.produced.len().saturating_sub(1));
                for j in 0..a {
                    rw.push(MtpRow {
                        slot: r.slot,
                        s: r.e() + j,
                        token: r.drafts[j as usize],
                        hidden: normed_at(i, r.p + j, &rows),
                    });
                }
            }
            if !rw.is_empty() {
                keep_alive.push(self.drafter(&rw)?);
            }
            let bs = self.cfg.block_size;
            for (i, r) in rows.iter().enumerate() {
                let a = n32(r.produced.len().saturating_sub(1));
                for q in r.c..=r.p + a {
                    if (q + 1).is_multiple_of(bs) {
                        let b = self.writable_block(r.slot, DRAFTER, q);
                        let pool = &self.pools[DRAFTER];
                        let tap = pool.taps.as_ref().expect("taps").addr
                            + u64::from(b) * pool.tap_bytes as u64;
                        // SAFETY: a normed row into the block's tap.
                        unsafe { self.dev.copy(tap, normed_at(i, q, &rows), hsz_us)? };
                    }
                }
                let keep: Vec<u32> = (r.p..=r.p + r.k).collect();
                for (j, &q) in keep.iter().enumerate() {
                    let dst = self.hidden_row(r.slot, j);
                    // SAFETY: a normed row into the slot's kept states.
                    unsafe { self.dev.copy(dst, normed_at(i, q, &rows), hsz_us)? };
                }
                self.slots[r.slot as usize].hiddens = keep;
            }
        }
        self.dev.synchronize()?;
        drop(keep_alive);

        for (i, r) in rows.iter().enumerate() {
            let (nb, nl) = new_base[i];
            let rec = &mut self.slots[r.slot as usize];
            rec.base = Some((nb, nl));
            rec.tail_len = r.e() + n32(r.drafts.len()) - nl;
            assert!(
                rec.tail_len <= self.tail_cap,
                "slot {}: a tail past its capacity",
                r.slot
            );
            rec.live_pos = (self.cfg.strategy == StateStrategy::Live).then_some(r.e());
        }

        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; rows.len() * stride as usize],
            stride,
            num_tokens: vec![0; rows.len()],
            logits: step.return_logits.then(Vec::new),
        };
        for (i, r) in rows.iter().enumerate() {
            let base = i * stride as usize;
            out.tokens[base..base + r.produced.len()].copy_from_slice(&r.produced);
            out.num_tokens[i] = n32(r.produced.len());
            if let Some(l) = out.logits.as_mut() {
                l.push(
                    (0..n32(r.produced.len()))
                        .map(|j| r.target_logits[&(r.p + j)].clone())
                        .collect(),
                );
            }
        }
        Ok(out)
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
        let (tail, tail_bytes) = self.slot_tail(slot);
        let k1 = self.cfg.draft_tokens as usize + 1;
        self.tables[s].iter().flatten().all(|&b| b == NULL_BLOCK)
            && self.slots[s].base.is_none()
            && self
                .read_words(self.live_addr(slot), self.image_stride)
                .iter()
                .all(|&x| x == 0)
            && self.read_words(tail, tail_bytes).iter().all(|&x| x == 0)
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
        for (i, b) in (1..=blocks).enumerate() {
            self.tables[slot as usize][TARGET][i] = b;
        }
        self.slots[slot as usize].base = Some((NULL_STATE, 0));
        let run = Run {
            slot,
            start: 0,
            tokens: tokens.to_vec(),
            src: 0,
            src_pos: 0,
            writes: Vec::new(),
            live_store: false,
            tail_from: u32::MAX,
            kv: true,
        };
        let f = self.target(std::slice::from_ref(&run))?;
        let hsz = u64::from(self.dims.hidden) * 2;
        let mut logits = Vec::with_capacity(tokens.len());
        // In pieces of 64 rows: the head's output is the whole vocabulary a row.
        for piece in (0..n).collect::<Vec<_>>().chunks(64) {
            let rows: Vec<u64> = piece
                .iter()
                .map(|&q| f.normed + u64::from(q) * hsz)
                .collect();
            logits.extend(self.head(&rows)?);
        }
        drop(f);
        for b in 1..=blocks {
            self.zero_block(0, b)?;
        }
        self.reset_slot(slot)?;
        self.dev.synchronize()?;
        Ok(logits)
    }
}
