//! The executor seam: the boundary between the host engine (scheduling, KV allocation,
//! prefix caching, lifecycle) and whatever runs the model (a CPU reference, a CUDA-graph
//! executor, or the deterministic mock).
//!
//! # Ownership
//!
//! * The **executor owns all device memory**: the physical KV pools (one per KV group, each
//!   `num_blocks` blocks of `block_size` positions across all of the group's layers), the
//!   persistent per-slot block tables, and per-slot model state (e.g. the hidden state an
//!   MTP drafter continues from, or a block drafter's target-layer taps).
//! * The **host decides everything about allocation**: which physical block holds which
//!   positions of which sequence, which state slot a sequence occupies, when a block is
//!   zeroed, copied, or reused. The executor never allocates, frees, or reinterprets.
//!
//! # One step
//!
//! [`Executor::execute`] runs exactly this sequence, with no host interaction in between:
//!
//! 1. `maintenance`, in order ([`Maintenance::Copy`], [`Maintenance::Zero`],
//!    [`Maintenance::ResetSlot`]). Every zero completes before any later write to that
//!    block, including writes by this step's forward.
//! 2. `table_updates`, in order: `table[slot][group][index] = block`.
//! 3. The forward: for each [`SeqEntry`], its host tokens occupy positions
//!    `context_len .. context_len + num_tokens`. KV for every query position is written
//!    to `table[slot][group][pos / block_size]` at offset `pos % block_size`, for every
//!    group. Queries attend per their group's [`AttentionKind`](crate::spec::AttentionKind)
//!    over positions `<= pos` of the same sequence.
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
//! # Graph capture
//!
//! Everything a step varies is data, not shape: the batch is padded to [`StepInput::bucket`]
//! (one captured graph per bucket), block tables live on the device indexed by slot and
//! change only through `table_updates`, sampling parameters are per-row arrays, and the
//! output is a fixed `[max_seqs, k + 1]` token array plus counts. Drafting, verification,
//! and acceptance run inside the step, so a speculative decode step needs no host round
//! trip. Steps with `num_drafts` uniform across rows and one host token per row (pure
//! decode) are the ones expected to replay a captured graph; mixed prefill steps may run
//! piecewise. `maintenance` and `table_updates` are applied by small kernels or copies
//! before the graph replays.
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
//! is completed later or that depends on a following token.
//!
//! # Drafter rows
//!
//! A drafter that consumes a token together with a hidden state from the previous
//! position must index its row by the **token it consumes**. For MTP depth `d` (level 0 is
//! the target's hidden state, level `d + 1` is depth `d`'s output):
//!
//! * the row at position `s` consumes the token at `s` and level `d` at `s - 1`, uses
//!   RoPE position `s - 1`, writes its KV at `s`, and predicts the token at `s + 1`; rows
//!   exist for `s >= d + 1`;
//! * the draft for position `p + 1 + i` (after the last host token `p`) is depth `i`'s
//!   prediction at `p + i`; depth `i`'s speculative rows occupy `p + 1 ..= p + i`, inside
//!   the positions the host reserved for drafts.
//!
//! Indexing a row by the hidden state it continues from instead (`(h_p, t_{p+1})` at `p`)
//! would make a block's last drafter row depend on the next block's first token, which
//! the block's cache key does not cover.
//!
//! A sequence resuming at a block boundary `c` (prefix hit or preemption resume) needs the
//! drafter's levels at `c - 1`, which no KV holds. The executor therefore stores a
//! **boundary tap** in each drafter block: the levels at the block's last position. Taps
//! are part of the block (a function of the tokens it covers), are zeroed and copied with
//! it, and a fresh slot loads its drafter state from the tap of the block ending at
//! `c - 1`; a running sequence carries it in per-slot state.
//!
//! A block drafter (DFlash shape) satisfies the same invariant: its per-position context
//! KV at `s` is projected from target hidden states at `s`, which depend only on tokens
//! `<= s`; the target taps it continues from at a resume point are stored the same way. A GPU executor that cannot make attention or GEMMs
//! batch-invariant must document its tolerance; the CPU reference executor meets it exactly.

use crate::sampling::SamplingParams;
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

/// Everything one step needs.
#[derive(Clone, Debug, PartialEq)]
pub struct StepInput {
    /// The captured shape this step is padded to.
    pub bucket: Bucket,
    /// Executed first, in order.
    pub maintenance: Vec<Maintenance>,
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
