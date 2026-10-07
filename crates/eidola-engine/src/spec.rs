//! The model spec an executor reports: everything the host engine needs to know about the
//! model and the executor's memory, and nothing about numerics.

use crate::sampling::Logits;

/// How a KV group's layers attend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    /// Every query sees every earlier position (global attention).
    Full,
    /// A query at position `q` sees keys at positions `k` with `q - window < k <= q`,
    /// i.e. itself and the `window - 1` positions before it. Learned attention-sink logits
    /// (a per-head extra softmax column) need no KV and do not change this.
    Sliding {
        /// Visible positions, including the query's own.
        window: u32,
    },
}

impl AttentionKind {
    /// First position a query at `q` can see.
    pub fn first_visible(self, q: u32) -> u32 {
        match self {
            AttentionKind::Full => 0,
            AttentionKind::Sliding { window } => (q + 1).saturating_sub(window),
        }
    }
}

/// Whose KV a group holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvRole {
    /// The target model's attention layers.
    Target,
    /// A speculative drafter's per-position context KV (an MTP head's sliding-window
    /// layers, or a block drafter's context projected from target hidden states). Like
    /// every group, each block depends only on the tokens it covers; see
    /// [`crate::executor`] for the row indexing that guarantees it.
    Drafter,
}

/// One KV group: a set of layers that share a block table and a physical block pool.
///
/// All layers of a group have the same attention kind and KV shape, so one physical block
/// id addresses the same token range in each of the group's per-layer tensors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvGroupSpec {
    /// Human-readable name (diagnostics only).
    pub name: String,
    /// Target or drafter.
    pub role: KvRole,
    /// Full or sliding-window attention.
    pub attention: AttentionKind,
    /// Layers in the group.
    pub num_layers: u32,
    /// KV heads per layer.
    pub num_kv_heads: u32,
    /// Head dimension of Q and K.
    pub head_dim_qk: u32,
    /// Head dimension of V.
    pub head_dim_v: u32,
    /// Physical blocks the executor allocated for this group, including the reserved null
    /// block 0. The host may hand out ids `1..num_blocks`.
    pub num_blocks: u32,
}

/// A captured step shape: the executor can run any batch with at most `max_seqs`
/// sequences and at most `max_tokens` query tokens (host tokens plus drafted tokens) as
/// one pre-built graph, after padding to exactly this shape.
///
/// Buckets have no ordering of their own: a lexicographic one would call a ladder
/// "sorted" that is not (see [`ModelSpec::buckets`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bucket {
    /// Sequence rows.
    pub max_seqs: u32,
    /// Query-token rows.
    pub max_tokens: u32,
}

/// The executor's description of the model and of the memory it owns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSpec {
    /// Logit rows the model computes, including padding rows past the tokenizer's last
    /// token (MiMo's head has 152,576 rows).
    pub vocab_size: u32,
    /// Token ids the model may emit: `0..sampleable_vocab_size`, every id the tokenizer
    /// defines (base vocabulary plus added tokens; 151,675 for MiMo-V2.6). Supplied by
    /// whoever loads the tokenizer, never inferred from the weights. Executors sample,
    /// draft and accept only over this prefix of each logit row ([`ModelSpec::logits`]),
    /// so padded ids are unsampleable by construction; the engine also refuses any
    /// returned id outside it.
    pub sampleable_vocab_size: u32,
    /// Tokens per KV block, shared by every group.
    pub block_size: u32,
    /// Longest sequence (prompt plus output) the executor supports.
    pub max_model_len: u32,
    /// KV groups. Group indices in the executor seam index this list.
    pub kv_groups: Vec<KvGroupSpec>,
    /// Drafted tokens per speculative step (`k`); 0 disables speculative decoding.
    pub max_draft_tokens: u32,
    /// Per-sequence device state rows (block tables, drafter hidden states); the maximum
    /// number of concurrently running sequences.
    pub num_state_slots: u32,
    /// Captured step shapes: a **ladder**, nondecreasing in both `max_seqs` and
    /// `max_tokens`, so the last bucket dominates every other and is the step capacity
    /// the scheduler plans against. Trade-off shapes (more sequences but fewer tokens)
    /// are refused by [`ModelSpec::validate`]: with them, no single bucket bounds a step.
    pub buckets: Vec<Bucket>,
}

/// Reserved physical block id in every group: never allocated, never written, never read.
/// Block-table entries the sequence does not own (out of window, beyond its length) hold it.
pub const NULL_BLOCK: u32 = 0;

impl ModelSpec {
    /// Blocks needed to hold `tokens` positions.
    pub fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.block_size)
    }

    /// Width of a per-slot block table: one entry per logical block of the longest sequence.
    pub fn max_blocks_per_seq(&self) -> u32 {
        self.blocks_for(self.max_model_len)
    }

    /// The sampleable part of a full logit row (every sampler input goes through this).
    /// Panics unless `row` has `vocab_size` entries.
    pub fn logits<'a>(&self, row: &'a [f32]) -> Logits<'a> {
        assert_eq!(
            row.len(),
            self.vocab_size as usize,
            "a logit row has vocab_size entries"
        );
        Logits::new(row, self.sampleable_vocab_size)
    }

    /// Smallest bucket that fits a batch, if any.
    pub fn bucket_for(&self, seqs: u32, tokens: u32) -> Option<Bucket> {
        self.buckets
            .iter()
            .copied()
            .filter(|b| b.max_seqs >= seqs && b.max_tokens >= tokens)
            .min_by_key(|b| (b.max_tokens, b.max_seqs))
    }

    /// Checks internal consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size == 0 || self.vocab_size == 0 || self.max_model_len == 0 {
            return Err("block_size, vocab_size and max_model_len must be non-zero".into());
        }
        if self.sampleable_vocab_size == 0 || self.sampleable_vocab_size > self.vocab_size {
            return Err(format!(
                "sampleable_vocab_size {} must be in 1..={}",
                self.sampleable_vocab_size, self.vocab_size
            ));
        }
        if self.kv_groups.is_empty() {
            return Err("at least one KV group is required".into());
        }
        if !self.kv_groups.iter().any(|g| g.role == KvRole::Target) {
            return Err("at least one target KV group is required".into());
        }
        for g in &self.kv_groups {
            if g.num_blocks < 2 {
                return Err(format!(
                    "group {} needs at least one allocatable block",
                    g.name
                ));
            }
            if let AttentionKind::Sliding { window } = g.attention
                && window == 0
            {
                return Err(format!("group {} has a zero window", g.name));
            }
        }
        if self.num_state_slots == 0 {
            return Err("num_state_slots must be non-zero".into());
        }
        if self.buckets.is_empty() {
            return Err("at least one bucket is required".into());
        }
        if self
            .buckets
            .windows(2)
            .any(|w| w[0].max_seqs > w[1].max_seqs || w[0].max_tokens > w[1].max_tokens)
        {
            return Err("buckets must be nondecreasing in both max_seqs and max_tokens".into());
        }
        Ok(())
    }
}
