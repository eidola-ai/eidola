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
//!
//! `Encrypt` and `Decrypt` really XOR a block's image (each entry's value as two
//! little-endian words, then its position) with the [`crate::kv_cipher`] keystream, so a
//! read of a block left encrypted fails like a read of a zeroed one. The mock also records
//! which key encrypted each block (as a keystream fingerprint, not the key) and fails a
//! step that decrypts under another key, decrypts a plaintext block or encrypts twice.
//!
//! With a host tier ([`KvGroupSpec::host_blocks`], see [`with_host_tier`]) the mock keeps
//! a second memory per group. `Demote` copies a block and its key fingerprint there and
//! fails on a block that is not encrypted; `Promote` copies back and fails unless the host
//! block holds ciphertext and the device block is zero (its zero ran first); `ZeroHost`
//! scrubs. So a plaintext block can never reach the host tier through a passing step.

use std::collections::HashMap;

mod chain;

pub use chain::Reference;

use crate::executor::{Executor, ExecutorError, Maintenance, SeqEntry, StepInput, StepOutput};
use crate::kv_cipher::{KvKeystream, KvNonce};
use crate::sampling::{self, SamplingParams, Stream, mix64};
use crate::secret::KvKey;
use crate::spec::{
    AttentionKind, Bucket, Coverage, KvGroupSpec, KvRole, ModelSpec, NULL_BLOCK, WindowChain,
};

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
        host_blocks: 0,
        ratio: 1,
        device: 0,
        coverage: Coverage::All,
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
        window_chain: None,
        token_history: 0,
    }
}

/// A small DeepSeek-V4.1-shaped spec, on two logical devices: the encoder's pooled
/// global bank (`ratio` positions per entry) and its sliding window on device 0; on
/// device 1 the decoder's per-position bank, its window chain of `depth` layers of
/// `window`, and a drafter whose context window is `window + 1` (the `window` positions
/// before a drafting row's own). Three token ids of history.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_like_spec(
    vocab: u32,
    block_size: u32,
    ratio: u32,
    window: u32,
    depth: u32,
    num_blocks: u32,
    slots: u32,
    k: u32,
) -> ModelSpec {
    let group = |name: &str, role, attention, layers, ratio, device, coverage| KvGroupSpec {
        name: name.into(),
        role,
        attention,
        num_layers: layers,
        num_kv_heads: 1,
        head_dim_qk: 512,
        head_dim_v: 512,
        num_blocks,
        host_blocks: 0,
        ratio,
        device,
        coverage,
    };
    let (target, all) = (KvRole::Target, Coverage::All);
    let sliding = |window| AttentionKind::Sliding { window };
    ModelSpec {
        kv_groups: vec![
            group(
                "encoder-bank",
                target,
                AttentionKind::Full,
                1,
                ratio,
                0,
                all,
            ),
            group("encoder-window", target, sliding(window), 1, 1, 0, all),
            group("decoder-bank", target, AttentionKind::Full, 1, 1, 1, all),
            group(
                "decoder-window",
                target,
                sliding(window),
                depth,
                1,
                1,
                Coverage::Chain,
            ),
            group(
                "drafter",
                KvRole::Drafter,
                sliding(window + 1),
                1,
                1,
                1,
                Coverage::Chain,
            ),
        ],
        max_draft_tokens: k,
        window_chain: Some(WindowChain { depth, window }),
        token_history: 3,
        max_model_len: 16384,
        ..mimo_like_spec(vocab, block_size, window, window, num_blocks, slots, k)
    }
}

/// `spec` with a host tier of `host_blocks` blocks (the null block included) in every
/// group.
pub fn with_host_tier(mut spec: ModelSpec, host_blocks: u32) -> ModelSpec {
    for g in &mut spec.kv_groups {
        g.host_blocks = host_blocks;
    }
    spec
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
    if spec.window_chain.is_some() {
        let mut dense = Reference::new(spec, cfg);
        for &t in &tokens[..=q as usize] {
            dense.push(t);
        }
        return dense.logits(q);
    }
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
    let mut dense = spec.window_chain.map(|_| {
        let mut dense = Reference::new(spec, cfg);
        for &t in prompt {
            dense.push(t);
        }
        dense
    });
    while (out.len() as u32) < max_tokens && (tokens.len() as u32) < spec.max_model_len {
        let q = tokens.len() as u32 - 1;
        let logits = match &dense {
            Some(dense) => dense.logits(q),
            None => reference_logits(spec, cfg, &tokens, q),
        };
        let t = sampling::sample(spec.logits(&logits), params, q as u64 + 1);
        if let Some(dense) = &mut dense {
            dense.push(t);
        }
        tokens.push(t);
        out.push(t);
        if stop.contains(&t) {
            break;
        }
    }
    out
}

/// A stand-in for "which key": the first keystream words under a fixed nonce no block
/// hash yields in practice. Test instrumentation only.
fn fingerprint(key: &KvKey) -> u64 {
    let mut w = [0u32; 2];
    KvKeystream::new(key, &KvNonce::from_words([u32::MAX; 3])).apply_words(&mut w);
    u64::from(w[0]) | (u64::from(w[1]) << 32)
}

/// The mock executor.
#[derive(Debug)]
pub struct MockExecutor {
    spec: ModelSpec,
    cfg: MockConfig,
    /// `[group][block * cells + cell] = (value, position)`; value 0 means zeroed or never
    /// written. A block has one cell per entry and layer ([`MockExecutor::cells`]).
    memory: Vec<Vec<(u64, u32)>>,
    /// `[group]`: cells per block.
    cells: Vec<usize>,
    /// The window-chain model's shape, when the spec has a chain.
    chain: Option<chain::Shape>,
    /// `[slot]`: the chain model's open pooled entries, as `(position, value)`.
    pairs: Vec<Vec<(u32, u64)>>,
    /// Every executed encrypt's nonce words as `(step, group, block, words)`.
    nonce_log: Vec<(u64, u32, u32, [u32; 3])>,
    /// Window KV cells the chain's layers wrote (layer-positions computed).
    chain_writes: u64,
    /// `[slot][group][index]`.
    tables: Vec<Vec<Vec<u32>>>,
    /// `[group]`: block -> number of (slot, index) mappings.
    mapped: Vec<HashMap<u32, u32>>,
    zero_log: Vec<(u64, u32, u32)>,
    /// `[group][block]`: the fingerprint of the key the block is encrypted under.
    encrypted: Vec<Vec<Option<u64>>>,
    /// Every executed encrypt (`true`) or decrypt as `(step, group, block, encrypt)`.
    crypt_log: Vec<(u64, u32, u32, bool)>,
    /// `[group][host * block_size + offset]`: the host tier, like `memory`.
    host_memory: Vec<Vec<(u64, u32)>>,
    /// `[group][host]`: the fingerprint of the key a host block's ciphertext is under.
    host_encrypted: Vec<Vec<Option<u64>>>,
    /// Every executed demote (`true`) or promote as `(step, group, block, host, demote)`.
    transfer_log: Vec<(u64, u32, u32, u32, bool)>,
    /// Every executed host zero as `(step, group, host)`.
    host_zero_log: Vec<(u64, u32, u32)>,
    steps: u64,
    buckets_used: Vec<Bucket>,
    rows_used: Vec<Vec<SeqEntry>>,
}

impl MockExecutor {
    /// A mock over `spec`.
    pub fn new(spec: ModelSpec, cfg: MockConfig) -> Self {
        let width = spec.max_blocks_per_seq() as usize;
        let groups = spec.kv_groups.len();
        let cells: Vec<usize> = spec
            .kv_groups
            .iter()
            .map(|g| {
                let layers = match (g.coverage, g.role) {
                    (Coverage::Chain, KvRole::Target) => g.num_layers,
                    _ => 1,
                };
                (spec.block_size / g.ratio * layers) as usize
            })
            .collect();
        Self {
            memory: spec
                .kv_groups
                .iter()
                .zip(&cells)
                .map(|(g, c)| vec![(0, 0); g.num_blocks as usize * c])
                .collect(),
            chain: spec.window_chain.map(|_| chain::Shape::of(&spec)),
            pairs: vec![Vec::new(); spec.num_state_slots as usize],
            nonce_log: Vec::new(),
            chain_writes: 0,
            tables: vec![vec![vec![NULL_BLOCK; width]; groups]; spec.num_state_slots as usize],
            mapped: vec![HashMap::new(); groups],
            zero_log: Vec::new(),
            encrypted: spec
                .kv_groups
                .iter()
                .map(|g| vec![None; g.num_blocks as usize])
                .collect(),
            crypt_log: Vec::new(),
            host_memory: spec
                .kv_groups
                .iter()
                .zip(&cells)
                .map(|(g, c)| vec![(0, 0); g.host_blocks as usize * c])
                .collect(),
            cells,
            host_encrypted: spec
                .kv_groups
                .iter()
                .map(|g| vec![None; g.host_blocks as usize])
                .collect(),
            transfer_log: Vec::new(),
            host_zero_log: Vec::new(),
            steps: 0,
            buckets_used: Vec::new(),
            rows_used: Vec::new(),
            spec,
            cfg,
        }
    }

    /// Every executed zero as `(step, group, block)`.
    pub fn zero_log(&self) -> &[(u64, u32, u32)] {
        &self.zero_log
    }

    /// Every executed encrypt (`true`) or decrypt as `(step, group, block, encrypt)`.
    pub fn crypt_log(&self) -> &[(u64, u32, u32, bool)] {
        &self.crypt_log
    }

    /// Whether a block is held encrypted.
    pub fn block_is_encrypted(&self, group: u32, block: u32) -> bool {
        self.encrypted[group as usize][block as usize].is_some()
    }

    /// The keystream fingerprint of the key a block is encrypted under (`None`: plaintext).
    /// Equal fingerprints mean the same key.
    pub fn encrypted_under(&self, group: u32, block: u32) -> Option<u64> {
        self.encrypted[group as usize][block as usize]
    }

    /// Every executed encrypt's nonce words as `(step, group, block, words)`.
    pub fn nonce_log(&self) -> &[(u64, u32, u32, [u32; 3])] {
        &self.nonce_log
    }

    /// Layer-positions the window chain computed so far.
    pub fn chain_writes(&self) -> u64 {
        self.chain_writes
    }

    /// Whether the chain model holds no open pooled entry for `slot`.
    pub fn slot_state_is_clear(&self, slot: u32) -> bool {
        self.pairs[slot as usize].is_empty()
    }

    /// Every executed demote (`true`) or promote as `(step, group, block, host, demote)`.
    pub fn transfer_log(&self) -> &[(u64, u32, u32, u32, bool)] {
        &self.transfer_log
    }

    /// Every executed host zero as `(step, group, host)`.
    pub fn host_zero_log(&self) -> &[(u64, u32, u32)] {
        &self.host_zero_log
    }

    /// Whether every byte of a host block is zero.
    pub fn host_block_is_zero(&self, group: u32, host: u32) -> bool {
        self.host_block(group, host)
            .iter()
            .all(|&(v, p)| v == 0 && p == 0)
    }

    /// The keystream fingerprint a host block's ciphertext is under (`None`: it holds
    /// none).
    pub fn host_encrypted_under(&self, group: u32, host: u32) -> Option<u64> {
        self.host_encrypted[group as usize][host as usize]
    }

    /// A host block's stored entries, `(value, position)` per offset, as they are.
    pub fn host_block(&self, group: u32, host: u32) -> &[(u64, u32)] {
        let bs = self.cells[group as usize];
        let start = host as usize * bs;
        &self.host_memory[group as usize][start..start + bs]
    }

    /// A device block's stored entries, `(value, position)` per offset, as they are.
    pub fn device_block(&self, group: u32, block: u32) -> &[(u64, u32)] {
        let bs = self.cells[group as usize];
        let start = block as usize * bs;
        &self.memory[group as usize][start..start + bs]
    }

    /// Steps executed.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Buckets of the executed steps, in order.
    pub fn buckets_used(&self) -> &[Bucket] {
        &self.buckets_used
    }

    /// Rows of the executed steps, in order.
    pub fn rows_used(&self) -> &[Vec<SeqEntry>] {
        &self.rows_used
    }

    /// Whether every byte of a block is zero.
    pub fn block_is_zero(&self, group: u32, block: u32) -> bool {
        self.device_block(group, block)
            .iter()
            .all(|&(v, p)| v == 0 && p == 0)
    }

    /// XORs block `block` of `group` with its keystream, and checks and records its state.
    fn crypt(
        &mut self,
        group: u32,
        block: u32,
        key: &KvKey,
        nonce: &KvNonce,
        encrypt: bool,
    ) -> Result<(), ExecutorError> {
        let print = fingerprint(key);
        let state = &mut self.encrypted[group as usize][block as usize];
        match (encrypt, *state) {
            (true, None) => *state = Some(print),
            (false, Some(p)) if p == print => *state = None,
            (false, Some(_)) => {
                return Err(ExecutorError(format!(
                    "group {group} block {block}: decrypt under another key"
                )));
            }
            (true, Some(_)) => {
                return Err(ExecutorError(format!(
                    "group {group} block {block}: encrypted twice"
                )));
            }
            (false, None) => {
                return Err(ExecutorError(format!(
                    "group {group} block {block}: decrypt of a plaintext block"
                )));
            }
        }
        let bs = self.cells[group as usize];
        let start = block as usize * bs;
        let cells = &mut self.memory[group as usize][start..start + bs];
        if encrypt {
            self.nonce_log
                .push((self.steps, group, block, nonce.words()));
        }
        KvKeystream::new(key, nonce).range(3 * bs, |i, k| {
            let (v, p) = &mut cells[i / 3];
            match i % 3 {
                0 => *v ^= u64::from(k),
                1 => *v ^= u64::from(k) << 32,
                _ => *p ^= k,
            }
        });
        self.crypt_log.push((self.steps, group, block, encrypt));
        Ok(())
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

    /// The cell of `layer` of the entry covering `pos` in the block `slot` maps there.
    fn cell(&self, slot: u32, group: u32, layer: u32, pos: u32) -> Result<usize, ExecutorError> {
        let gs = &self.spec.kv_groups[group as usize];
        let bs = self.spec.block_size;
        let block = self.tables[slot as usize][group as usize][(pos / bs) as usize];
        if block == NULL_BLOCK {
            return Err(ExecutorError(format!(
                "slot {slot} group {group} position {pos}: unmapped block"
            )));
        }
        let cells = self.cells[group as usize];
        let layers = cells / (bs / gs.ratio) as usize;
        Ok(block as usize * cells + (pos % bs / gs.ratio) as usize * layers + layer as usize)
    }

    /// Writes `layer`'s value for the entry completed at `pos`, from logical device `dev`.
    fn put(
        &mut self,
        slot: u32,
        group: u32,
        layer: u32,
        pos: u32,
        value: u64,
        dev: u32,
    ) -> Result<(), ExecutorError> {
        if self.spec.kv_groups[group as usize].device != dev {
            return Err(ExecutorError(format!(
                "group {group} written from device {dev}"
            )));
        }
        let at = self.cell(slot, group, layer, pos)?;
        let block =
            self.tables[slot as usize][group as usize][(pos / self.spec.block_size) as usize];
        if self.mapped[group as usize]
            .get(&block)
            .copied()
            .unwrap_or(0)
            != 1
        {
            return Err(ExecutorError(format!(
                "slot {slot} group {group} position {pos}: write to a shared block"
            )));
        }
        self.memory[group as usize][at] = (value, pos);
        Ok(())
    }

    /// Reads `layer`'s value for the entry completed at `pos`, from logical device `dev`;
    /// fails where that layer never wrote it.
    fn get(
        &self,
        slot: u32,
        group: u32,
        layer: u32,
        pos: u32,
        dev: u32,
    ) -> Result<u64, ExecutorError> {
        if self.spec.kv_groups[group as usize].device != dev {
            return Err(ExecutorError(format!(
                "group {group} read from device {dev}"
            )));
        }
        let at = self.cell(slot, group, layer, pos)?;
        let (v, p) = self.memory[group as usize][at];
        if v == 0 || p != pos {
            return Err(ExecutorError(format!(
                "slot {slot} group {group} layer {layer} position {pos}: read of KV that \
                 was never computed, zeroed or foreign"
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
        Ok(self.mix_draft(noise_state, target, params))
    }

    /// The drafter's distribution: the target's mixed with noise seeded by `noise_state`,
    /// by the configured agreement; its argmax, one-hot, for a greedy row.
    fn mix_draft(&self, noise_state: u64, target: &[f64], params: &SamplingParams) -> Vec<f64> {
        let noise_logits = logits_from_state(&self.cfg, self.spec.vocab_size, noise_state ^ 0x77);
        let noise = sampling::processed_probs(
            self.spec.logits(&noise_logits),
            &SamplingParams::random(1.0, 0).expect("valid constant"),
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
            one_hot
        } else {
            mix
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
        self.rows_used.push(step.seqs.clone());
        for m in &step.maintenance {
            let bs = match *m {
                Maintenance::Copy { group, .. }
                | Maintenance::Zero { group, .. }
                | Maintenance::Encrypt { group, .. }
                | Maintenance::Decrypt { group, .. }
                | Maintenance::Demote { group, .. }
                | Maintenance::Promote { group, .. }
                | Maintenance::ZeroHost { group, .. } => self.cells[group as usize],
                Maintenance::ResetSlot { .. } => 0,
            };
            match *m {
                Maintenance::Zero { group, block } => {
                    let start = block as usize * bs;
                    self.memory[group as usize][start..start + bs].fill((0, 0));
                    self.encrypted[group as usize][block as usize] = None;
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
                    let key = step.kv_keys.get(key as usize).ok_or_else(|| {
                        ExecutorError(format!("maintenance names key {key}, not in the step"))
                    })?;
                    self.crypt(group, block, key, &nonce, encrypt)?;
                }
                Maintenance::Copy { group, src, dst } => {
                    let g = &mut self.memory[group as usize];
                    g.copy_within(
                        src as usize * bs..(src as usize + 1) * bs,
                        dst as usize * bs,
                    );
                    let e = &mut self.encrypted[group as usize];
                    e[dst as usize] = e[src as usize];
                }
                Maintenance::Demote { group, block, host } => {
                    let (g, b, h) = (group as usize, block as usize, host as usize);
                    let Some(print) = self.encrypted[g][b] else {
                        return Err(ExecutorError(format!(
                            "group {group} block {block}: demote of a plaintext block"
                        )));
                    };
                    let cells = self.memory[g][b * bs..(b + 1) * bs].to_vec();
                    self.host_memory[g][h * bs..(h + 1) * bs].copy_from_slice(&cells);
                    self.host_encrypted[g][h] = Some(print);
                    self.transfer_log
                        .push((self.steps, group, block, host, true));
                }
                Maintenance::Promote { group, host, block } => {
                    let (g, b, h) = (group as usize, block as usize, host as usize);
                    let Some(print) = self.host_encrypted[g][h] else {
                        return Err(ExecutorError(format!(
                            "group {group} host block {host}: promote of a block holding no ciphertext"
                        )));
                    };
                    if !self.block_is_zero(group, block) || self.encrypted[g][b].is_some() {
                        return Err(ExecutorError(format!(
                            "group {group} block {block}: promote into a block that is not zero"
                        )));
                    }
                    let cells = self.host_memory[g][h * bs..(h + 1) * bs].to_vec();
                    self.memory[g][b * bs..(b + 1) * bs].copy_from_slice(&cells);
                    self.encrypted[g][b] = Some(print);
                    self.transfer_log
                        .push((self.steps, group, block, host, false));
                }
                Maintenance::ZeroHost { group, host } => {
                    let (g, h) = (group as usize, host as usize);
                    self.host_memory[g][h * bs..(h + 1) * bs].fill((0, 0));
                    self.host_encrypted[g][h] = None;
                    self.host_zero_log.push((self.steps, group, host));
                }
                Maintenance::ResetSlot { slot } => {
                    self.pairs[slot as usize].clear();
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
            if self.chain.is_some() {
                let (produced, row_logits) = self.chain_row(step, row, e)?;
                let base = row * stride as usize;
                out.tokens[base..base + produced.len()].copy_from_slice(&produced);
                out.num_tokens[row] = produced.len() as u32;
                if let Some(l) = out.logits.as_mut() {
                    l.push(row_logits);
                }
                continue;
            }
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
                            sampling::uniform(params.seed(), q as u64 + 1, Stream::Draft),
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
