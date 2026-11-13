//! The executor seam: the boundary between the host engine (scheduling, KV allocation,
//! prefix caching, lifecycle) and whatever runs the model (a CPU reference, a CUDA-graph
//! executor, or the deterministic mock).
//!
//! # Ownership
//!
//! * The **executor owns all device memory**: the physical KV pools (one per KV group, each
//!   `num_blocks` blocks of `block_size` positions across all of the group's layers), the
//!   persistent per-slot block tables, and per-slot model state (e.g. the hidden state an
//!   MTP drafter continues from, a block drafter's target-layer taps, or the open entry
//!   of a pooled group).
//! * The **host decides everything about allocation**: which physical block holds which
//!   positions of which sequence, which state slot a sequence occupies, when a block is
//!   zeroed, copied, or reused. The executor never allocates, frees, or reinterprets.
//!
//! # One step
//!
//! [`Executor::execute`] runs exactly this sequence, with no host interaction in between:
//!
//! 1. `maintenance`, in the order given ([`Maintenance::Copy`], [`Maintenance::Zero`],
//!    [`Maintenance::ResetSlot`], [`Maintenance::Encrypt`], [`Maintenance::Decrypt`],
//!    and the host tier's [`Maintenance::Demote`], [`Maintenance::Promote`] and
//!    [`Maintenance::ZeroHost`]). Every operation on a block takes effect after the
//!    operations listed before it that touch the same block (device or host) and before
//!    those after it: every zero before any later write to that block, every demote's read
//!    before the zero that follows it, every promote before the decrypt that follows it,
//!    and every decrypt before any read of the block, including this step's forward.
//! 2. `table_updates`, in order: `table[slot][group][index] = block`.
//! 3. The forward: for each [`SeqEntry`], its host tokens occupy positions
//!    `context_len .. context_len + num_tokens`. KV for every query position is written
//!    to `table[slot][group][pos / block_size]` at offset `pos % block_size`, for every
//!    group. Queries attend per their group's [`AttentionKind`](crate::spec::AttentionKind)
//!    over positions `<= pos` of the same sequence. (A pooled group writes one entry per
//!    `ratio` positions, and a chain-covered group only where the window chain computes:
//!    *Pooled groups* and *Window chain*, below.)
//! 4. If `sample` is set, the executor samples at the last host token
//!    (position `p = context_len + num_tokens - 1`, choosing the token at `p + 1`), on
//!    the device, with the [`crate::sampling`] semantics. If `num_drafts = k > 0` as well,
//!    it first lets its drafter propose `k` tokens for positions `p + 1 ..= p + k`, runs the
//!    target over them in the same pass (writing their KV to the slots the host reserved),
//!    and applies [`crate::sampling::chain_accept`]. The result is `1..=k + 1` tokens.
//!
//!    **No drafts at position 0.** A drafter's first proposal continues from the state at
//!    `p - 1` (MTP depth 0's row at `p` consumes the target's hidden state there), so a
//!    row whose last host position is `p = 0` has nothing to draft from. The host never
//!    gives such a row drafts, and an executor treats a request for them as a host bug
//!    rather than quietly sampling plainly: KV reserved for drafts that cannot exist would
//!    be wasted, and the draft counts would lie.
//!
//!    **Only sampleable ids.** Target sampling, the drafter's proposals, acceptance and the
//!    residual all run over the first [`ModelSpec::sampleable_vocab_size`] entries of each
//!    logit row ([`ModelSpec::logits`]); the padded rows past the tokenizer's last token
//!    take no part in any distribution, so no step can return, draft or feed back a padded
//!    id. A device sampler applies the same limit (it is part of the `sampling` semantics,
//!    not a separate mask), and the engine refuses a step that returns an id outside it.
//!
//! A step reads only the KV of its own sequences' slots and never writes a block it was not
//! directed to by the tables. Blocks the host shares between sequences (prefix-cache hits)
//! are never written: the host never schedules a query position inside a shared block.
//!
//! # Idle encryption
//!
//! A prefix-cache block that no sequence maps (an *idle* entry's block) is held encrypted
//! ([`crate::kv_cipher`]: ChaCha20 under the lineage's [`KvKey`], with the nonce the host
//! derived from the block's hash and group). The host issues [`Maintenance::Encrypt`]
//! when the last sequence mapping a cached block lets go of it and the cache keeps it,
//! and [`Maintenance::Decrypt`] before a sequence maps an encrypted block again; the keys
//! those operations name travel in [`StepInput::kv_keys`] for that step only. The
//! executor encrypts every byte of the block that [`Maintenance::Zero`] would zero (all
//! of the group's layers, and a drafter block's boundary tap), as the ranges of an image
//! whose order it documents; a block's representation (quantised KV, index keys beside
//! the KV, a pooled group's fewer entries) changes its bytes and nothing else. The nonce
//! is whatever the operation names: the host derives a chain-covered block's from its
//! attachment to a cache entry as well as its hash and group, because such a block's
//! bytes are not a function of the tokens it covers. Encrypting is an XOR with the keystream, so decrypting is
//! the same operation; they are distinct operations so that each can be checked.
//!
//! Between two steps the host names a block in at most one encrypt or decrypt, and never
//! lists a zero, copy or demote of a block after its encrypt or decrypt, so an executor
//! may run a run of consecutive encryptions and decryptions together (one launch); it must
//! still finish them before the forward reads. (A block may be zeroed, promoted into, and
//! then decrypted in one step: a demoted block's id reused for a promoted one.)
//! An executor keeps no key past the step: whatever device or host copy it makes of
//! [`StepInput::kv_keys`] is zeroed before `execute` returns or by stream-ordered work
//! queued before the forward.
//!
//! # Host tier
//!
//! An executor may keep a second, larger pool per group in host memory
//! ([`crate::spec::KvGroupSpec::host_blocks`]: the confidential VM's own memory, never
//! durable media), where the host parks idle prefix-cache blocks evicted from the device
//! pools under pressure instead of dropping them. Host blocks have their own ids
//! (`1..host_blocks`, 0 reserved), and the host owns their allocation exactly as it owns
//! the device pools'. Only ciphertext ever moves there:
//!
//! * [`Maintenance::Demote`] copies a device block's whole image (every byte
//!   [`Maintenance::Zero`] would zero) into a host block. The host demotes only a block
//!   that is encrypted when the step begins, and queues the device block's zero after it.
//! * [`Maintenance::Promote`] copies a host block's image back into a device block the
//!   host just allocated (and whose zero, if any, is listed before it). The block is still
//!   ciphertext; a [`Maintenance::Decrypt`] after it is what a hit then maps.
//! * [`Maintenance::ZeroHost`] overwrites a host block with zeros. The host issues it for
//!   every host block that stops holding an entry, after any promote that reads it, and
//!   never names that block in a demote before the next step. Unlike device work it may
//!   run after the forward, but it completes before `execute` returns.
//!
//! The executor never decrypts, encrypts or reads a host block for any other purpose, and
//! holds no key for one. Host-tier memory must be locked into RAM (page-locked, never
//! swapped); the node refuses to boot on a guest with active swap.
//!
//! # Graph capture
//!
//! Everything a step varies is data, not shape: the batch is padded to [`StepInput::bucket`]
//! (one captured graph per bucket), block tables live on the device indexed by slot and
//! change only through `table_updates`, sampling parameters are per-row arrays, and the
//! output is a fixed `[max_seqs, k + 1]` token array plus counts. Drafting, verification,
//! and acceptance run inside the step, so a speculative decode step needs no host round
//! trip. Steps with `num_drafts` uniform across rows and one host token per row (pure
//! decode) are the ones expected to replay a captured graph; mixed prefill steps may
//! run piecewise. The host gives every drafting row of a step the same width: the full
//! `k`, or none for every row of a step that ran short of KV (and none at position 0),
//! trimming only near the model length, so non-uniform decode steps are rare and need no
//! graph of their own. `maintenance` and `table_updates` are applied by small kernels or
//! copies before the graph replays. With a window chain, a step replays a decode graph
//! only if every row's `chain_start` equals its `context_len` (every layer computes
//! every position), which the host's data says before the step and which holds for
//! every row that samples from one host token.
//!
//! # Determinism contract
//!
//! For a given model, the KV written for a position and the logits produced for a query
//! must depend only on the sequence's tokens at positions `<= pos`, never on batch
//! composition, chunking, slot, or physical block ids. The host's preemption and
//! prefix-cache correctness rest on this.
//!
//! In particular, **every block in every KV group depends only on the tokens that block
//! covers** (and those before it): the host seals a block into the prefix cache as soon as
//! it is full and shares it with any sequence whose salted token chain matches up to the
//! block's end, for target and drafter groups alike. There is no allowance for KV that
//! is completed later or that depends on a following token. This is the contract of
//! every group with [`Coverage::All`](crate::spec::Coverage); a group with
//! [`Coverage::Chain`](crate::spec::Coverage) holds valid KV only where the window chain
//! computed, and the host never shares its blocks as sealed blocks (*Window chain*).
//!
//! # Pooled groups
//!
//! A full-attention group with [`KvGroupSpec::ratio`](crate::spec::KvGroupSpec::ratio)
//! `r > 1` holds one entry per `r` positions. Entry `j` is a function of the tokens at
//! positions `<= r * j + r - 1`, is written when a row computes that position, into
//! block `r * j / block_size` at entry offset `j % (block_size / r)`, and is visible to a
//! query at `q` exactly when `r * j + r - 1 <= q`. `r` divides the block size, so a
//! block's entries depend only on the tokens it covers and those before it, and a full
//! block is final: the block invariant holds as at ratio 1, with no boundary tap.
//!
//! What a row leaves when it ends inside an entry (the contributions of the positions
//! computed so far) is per-slot state: a function of the token prefix, kept for each
//! speculative position until the next step fixes the accepted length, zeroed by
//! [`Maintenance::ResetSlot`], and never needed at a resume, which starts at a block
//! boundary, where no entry is open.
//!
//! # Window chain
//!
//! A model with a [`WindowChain`](crate::spec::WindowChain) of depth `D` and window `W`
//! has a stack of sliding-window layers (the layers of its target group with
//! [`Coverage::Chain`](crate::spec::Coverage)) in which layer `j`'s window keys at a
//! position are a function of layer `j - 1`'s output there (for `j = 0`, of what runs
//! below the chain), and in which everything older than a layer's window reaches it only
//! through groups with [`Coverage::All`](crate::spec::Coverage). The chain runs only
//! where its results can matter. Each row carries `s`, its
//! [`StepAux::chain_start`], and:
//!
//! * chain layer `j` (0 = bottom) **computes** exactly the row's positions
//!   `x >= s - (D - 1 - j) * (W - 1)`, saturating at 0
//!   ([`ModelSpec::chain_layer_start`]), host tokens and drafts alike: it writes its
//!   window KV at `x`;
//! * layer `j`'s output at `x` is defined only where layer `j + 1` computes, and the top
//!   layer's (the hidden state and logits) only where the row samples or verifies a
//!   draft; an executor reads no window KV for an output nothing consumes;
//! * every other [`Coverage::Chain`](crate::spec::Coverage) group (a drafter's context
//!   KV, projected from the inputs of the chain's layers) is written where the top layer
//!   computes, `x >= s`;
//! * everything else runs at every position of the row.
//!
//! The host chooses `s` so that every window KV a consumed output reads has been
//! written, by this row, by an earlier one, or in blocks a prefix-cache hit mapped. It
//! plans toward **targets**, the positions that need full-depth state: a sequence's last
//! known token, every position generated after it, and the last position before each
//! boundary at which the sequence will retain its windows. With `top` the widest
//! chain-covered window less one ([`ModelSpec::chain_top`]), a target `t` needs the top
//! layer from `t - top` (its own window, for the logits at `t`; a drafter's context,
//! for a row that drafts at `t`), and each layer below from `W - 1` earlier: layer `j`
//! from `t - top - (D - 1 - j) * (W - 1)`, the bottom one from `t` less
//! [`ModelSpec::chain_reach`]. A row never extends past its target, and `s` is
//! `t - top`, or `context_len` when that is no later, or when the chain is valid at
//! every layer before the row and the bottom layer would have to start within it.
//!
//! A row that samples has its last host position for target, so a row that samples
//! from one host token has `s == context_len`: every layer, every position, drafts
//! included.
//!
//! The results are bit-identical to computing every layer at every position: which
//! cells are computed is a function of the targets and of where the sequence was
//! admitted, never of chunking, and every computed cell holds the value the full
//! computation gives. So the first paragraph of this contract holds for logits
//! unchanged. A chain-covered block's contents are not a function of the tokens it
//! covers; the host caches such blocks only as the windows attached at retained hit
//! points, valid at every layer there, and a hit maps every window of its point.
//!
//! # Token history
//!
//! A model with [`ModelSpec::token_history`] `h > 0` reads the `h` token ids before a
//! position beside the position's own (a lookup addressed by n-grams of ids). The host
//! passes them for each row's first host token in [`StepAux::history_ids`]; past it the
//! executor has the row's host tokens and its own drafts. No slot holds them, so a
//! resumed row needs nothing but KV.
//!
//! # Drafter rows
//!
//! A drafter that consumes a token together with a hidden state from an earlier position
//! must index its row by the **token it consumes**. For MTP depth `d`, every depth
//! conditioned on the **target's** hidden state (as the model vendor serves its MTP
//! layers: layer `d` combines `h_x` with the token `d + 1` past `x`, never another MTP
//! layer's output):
//!
//! * the row at position `s` consumes the token at `s` and the target's hidden state at
//!   the anchor `a = s - 1 - d`, uses RoPE position `a`, writes its KV at `s`, and
//!   predicts the token at `s + 1`; rows exist for `s >= d + 1`;
//! * the draft for position `p + 1 + i` (after the last host token `p`) is depth `i`'s
//!   prediction at `p + i`, anchored at `p - 1` like every depth's; depth `i`'s
//!   speculative rows occupy `p + 1 ..= p + i`, inside the positions the host reserved
//!   for drafts.
//!
//! A depth's output feeds only its own logits and KV. Level `l` at position `x` names the
//! target's state at `x - l` (zeros before position 0), so depth `d`'s row at `s` reads
//! level `d` at `s - 1`.
//!
//! Indexing a row by the hidden state it continues from instead (`(h_p, t_{p+1})` at `p`)
//! would make a block's last drafter row depend on the next block's first token, which
//! the block's cache key does not cover.
//!
//! A sequence resuming at a block boundary `c` (prefix hit or preemption resume) needs the
//! drafter's levels at `c - 1` (the target's states at `c - 1 ..= c - D` for `D` depths),
//! which no KV holds. The executor therefore stores a **boundary tap** in each drafter
//! block: the levels at the block's last position. Taps
//! are part of the block (a function of the tokens it covers), are zeroed and copied with
//! it, and a fresh slot loads its drafter state from the tap of the block ending at
//! `c - 1`; a running sequence carries it in per-slot state.
//!
//! A block drafter (DFlash shape) satisfies the same invariant: its per-position context
//! KV at `s` is projected from target hidden states at `s`, which depend only on tokens
//! `<= s`; the target taps it continues from at a resume point are stored the same way.
//!
//! A block drafter whose first slot takes the token at the row's last host position `p`
//! fits the same indexing: it continues from the taps at `p - 1`, reads its context KV
//! at positions before `p` (its group's window counts `p` itself, which it does not
//! read), and proposes `p + 1 ..= p + k`. Where a slot's distribution depends on the
//! token drafted for the slot before (a Markov term), the executor samples the slots
//! left to right on the draft stream and hands [`crate::sampling::chain_accept`], as
//! each slot's distribution, the conditional it sampled from; that is what its residual
//! assumes. Context KV is written from the taps of every position the pass computes,
//! drafts included, and a rejected position's is rewritten by the next step before
//! anything reads it. Whatever else a drafter reports (a confidence per slot) the seam
//! does not carry: every row verifies all `k`.
//!
//! Every executor meets this exactly, bit for bit: the CPU reference executor by
//! construction, the CUDA executor by a batch-invariant forward (one reduction order per
//! row, whatever the step holds: `eidola-engine-cuda/AGENTS.md` → Batch invariance).

use crate::kv_cipher::KvNonce;
use crate::sampling::SamplingParams;
use crate::secret::KvKey;
use crate::spec::{Bucket, ModelSpec};

/// A state slot: one row of the executor's per-sequence device tables.
pub type Slot = u32;

/// Memory maintenance, executed before the forward in the order given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Maintenance {
    /// Copy every layer's KV of `src` into `dst` within `group`.
    Copy {
        /// KV group index.
        group: u32,
        /// Source block.
        src: u32,
        /// Destination block.
        dst: u32,
    },
    /// Overwrite every byte of `block` in every layer of `group` with zeros. Issued for every
    /// physical block that stops holding live data (freed by a sequence or evicted from the
    /// prefix cache), before it can be handed to anyone else.
    Zero {
        /// KV group index.
        group: u32,
        /// Block to scrub.
        block: u32,
    },
    /// Reset a state slot: every block-table entry becomes the null block and any
    /// per-sequence model state (drafter hidden states, taps) is zeroed. Issued when a slot
    /// is released and before it is reused.
    ResetSlot {
        /// Slot to reset.
        slot: Slot,
    },
    /// Encrypt `block` of `group` in place: XOR every byte [`Maintenance::Zero`] would
    /// zero with the ChaCha20 keystream of `kv_keys[key]` and `nonce`
    /// ([`crate::kv_cipher`]). Issued when a cached block becomes idle.
    Encrypt {
        /// KV group index.
        group: u32,
        /// Block to encrypt.
        block: u32,
        /// Index into [`StepInput::kv_keys`].
        key: u32,
        /// The block's nonce.
        nonce: KvNonce,
    },
    /// Decrypt `block` of `group` in place (the same keystream XOR as
    /// [`Maintenance::Encrypt`]). Issued before an idle block is mapped again.
    Decrypt {
        /// KV group index.
        group: u32,
        /// Block to decrypt.
        block: u32,
        /// Index into [`StepInput::kv_keys`].
        key: u32,
        /// The block's nonce.
        nonce: KvNonce,
    },
    /// Copy every byte of device `block` that [`Maintenance::Zero`] would zero into host
    /// block `host` of `group`'s host tier. The block is ciphertext. Issued when an idle
    /// entry is demoted under device pressure; its zero follows.
    Demote {
        /// KV group index.
        group: u32,
        /// Device block to copy from.
        block: u32,
        /// Host block to copy into.
        host: u32,
    },
    /// Copy host block `host` of `group` back into device `block` (still ciphertext).
    /// Issued when a prefix-cache hit needs a demoted block; a decrypt follows if a
    /// sequence maps it.
    Promote {
        /// KV group index.
        group: u32,
        /// Host block to copy from.
        host: u32,
        /// Device block to copy into.
        block: u32,
    },
    /// Overwrite every byte of host block `host` of `group` with zeros. Issued for every
    /// host block that stops holding an entry (promoted, evicted or expired).
    ZeroHost {
        /// KV group index.
        group: u32,
        /// Host block to scrub.
        host: u32,
    },
}

/// One block-table entry assignment: `table[slot][group][index] = block`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableUpdate {
    /// State slot.
    pub slot: Slot,
    /// KV group index.
    pub group: u32,
    /// Logical block index within the sequence.
    pub index: u32,
    /// Physical block, or [`crate::spec::NULL_BLOCK`] to unmap.
    pub block: u32,
}

/// One sequence's row in a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeqEntry {
    /// The sequence's state slot.
    pub slot: Slot,
    /// Offset of this row's host tokens in [`StepInput::token_ids`].
    pub token_start: u32,
    /// Host tokens in this row (`>= 1`).
    pub num_tokens: u32,
    /// Positions already in KV before this step (the first host token's position).
    pub context_len: u32,
    /// Tokens the executor drafts and verifies after the host tokens; the host has
    /// reserved KV for them. Only meaningful with `sample`, and always 0 when the last
    /// host position is 0 (see the module docs: a drafter continues from the position
    /// before it, and position 0 has none).
    pub num_drafts: u32,
    /// Whether this row produces tokens (false for a prefill chunk that does not reach the
    /// end of the known tokens).
    pub sample: bool,
    /// Sampling parameters.
    pub sampling: SamplingParams,
}

/// Per-row inputs only some models need, row-aligned with [`StepInput::seqs`]. Both are
/// empty for a model with no window chain and no token history.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepAux {
    /// Per row, where the window chain's top layer starts (see *Window chain* in the
    /// module docs). Equal to the row's `context_len` when every layer computes every
    /// position of the row, which is always so for a row that samples from one host token. Empty unless
    /// the model has a [`crate::spec::WindowChain`].
    pub chain_start: Vec<u32>,
    /// Per row, [`ModelSpec::token_history`] token ids: the tokens at the positions just
    /// before `context_len`, oldest first, zero where that position is before the
    /// sequence's start.
    pub history_ids: Vec<u32>,
}

/// Everything one step needs.
#[derive(Clone, Debug, PartialEq)]
pub struct StepInput {
    /// The captured shape this step is padded to.
    pub bucket: Bucket,
    /// Executed first, in order.
    pub maintenance: Vec<Maintenance>,
    /// The keys this step's [`Maintenance::Encrypt`] and [`Maintenance::Decrypt`] name,
    /// by index; each zeroed when the step is dropped.
    pub kv_keys: Vec<KvKey>,
    /// Executed second, in order.
    pub table_updates: Vec<TableUpdate>,
    /// Sequence rows (at most `bucket.max_seqs`).
    pub seqs: Vec<SeqEntry>,
    /// Host tokens of all rows, concatenated in row order.
    pub token_ids: Vec<u32>,
    /// Position of each host token (redundant with `context_len`, provided flat for the
    /// device).
    pub positions: Vec<u32>,
    /// Testing aid: also return the target logits row for every produced token.
    pub return_logits: bool,
    /// Per-row inputs for a window chain and a token history; empty otherwise.
    pub aux: StepAux,
}

impl StepInput {
    /// Query tokens including drafts (what counts against `bucket.max_tokens`).
    pub fn query_tokens(&self) -> u32 {
        self.seqs.iter().map(|s| s.num_tokens + s.num_drafts).sum()
    }
}

/// A step's results, row-aligned with [`StepInput::seqs`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepOutput {
    /// Row-major `[seqs.len(), stride]` produced tokens.
    pub tokens: Vec<u32>,
    /// Row stride of `tokens`: `max_draft_tokens + 1`.
    pub stride: u32,
    /// Tokens produced per row (0 for rows without `sample`).
    pub num_tokens: Vec<u32>,
    /// When requested: per row, one target logits row per produced token.
    pub logits: Option<Vec<Vec<Vec<f32>>>>,
}

impl StepOutput {
    /// The tokens produced for row `i`.
    pub fn row(&self, i: usize) -> &[u32] {
        let start = i * self.stride as usize;
        &self.tokens[start..start + self.num_tokens[i] as usize]
    }
}

/// An executor failure. Fatal to the engine instance: the host cannot know which writes
/// landed.
#[derive(Debug)]
pub struct ExecutorError(pub String);

impl std::fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "executor failure: {}", self.0)
    }
}

impl std::error::Error for ExecutorError {}

/// Runs the model. See the module documentation for the full contract.
pub trait Executor {
    /// The model and memory description. Constant for the executor's lifetime.
    fn spec(&self) -> &ModelSpec;

    /// Executes one step.
    fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError>;
}
