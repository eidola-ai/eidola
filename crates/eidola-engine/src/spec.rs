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

/// Which positions of a group hold valid KV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Every computed position: each block depends only on the tokens it covers and those
    /// before it, so a full block is final and shareable.
    All,
    /// Only the positions the window chain computed ([`WindowChain`], and the row rule in
    /// [`crate::executor`]): a block's contents depend on which rows ran how deep, so it
    /// is never shared as a sealed block. The host caches such a group only as the windows
    /// attached at retained hit points, where it is valid at every layer.
    Chain,
}

/// A stack of sliding-window layers in which each layer's window keys at a position are a
/// function of the layer below's output there, run only where its results can matter. The
/// layers are those of the target group with [`Coverage::Chain`]; the row rule that says
/// which layer computes which position is in [`crate::executor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowChain {
    /// Layers in the chain, bottom first.
    pub depth: u32,
    /// Each layer's window: visible positions, including the query's own.
    pub window: u32,
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
    /// Blocks of this group the executor keeps in host memory for idle prefix-cache
    /// entries demoted from `num_blocks` (the host tier), including a reserved null
    /// block 0; 0 when there is no host tier. Every group has a host tier or none does.
    pub host_blocks: u32,
    /// Positions per entry (1: one entry per position). A block holds
    /// `block_size / ratio` entries; entry `j` is pooled from positions
    /// `ratio * j ..= ratio * j + ratio - 1`, is written when the last of them is
    /// computed, and is visible to a query at `q` once `q` has reached that position.
    /// `ratio` divides the block size, so no entry straddles a block. Only
    /// full-attention groups pool.
    pub ratio: u32,
    /// The logical device holding this group's pool (0 for a single-device model). The
    /// host only accounts memory by it; what crosses between devices is the executor's.
    pub device: u32,
    /// Which positions hold valid KV.
    pub coverage: Coverage,
}

impl KvGroupSpec {
    /// Entries of this group visible to a query at `q`, from the group's first:
    /// full attention sees every completed entry.
    pub fn visible_entries(&self, q: u32) -> u32 {
        (q + 1) / self.ratio
    }
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
    /// The window chain, for a model whose upper layers run only where needed; `None`
    /// for a model that runs every layer at every position.
    pub window_chain: Option<WindowChain>,
    /// Token ids before a row's first host token the executor needs beside KV (a
    /// lookup addressed by the current token and the ones before it); 0 for none. The
    /// host passes them with every row ([`crate::executor::StepAux::history_ids`]), so
    /// no slot holds them.
    pub token_history: u32,
}

/// Reserved physical block id in every group: never allocated and never written, so it
/// reads as zeros. Block-table entries the sequence does not own (out of window, beyond its
/// length) hold it. An executor may read it only where every key it supplies is masked (the
/// CUDA executor lists it for positions before a sliding window's anchor); no seam
/// operation reads or writes it.
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

    /// How far behind a target the chain's top layer starts: the widest window among
    /// the [`Coverage::Chain`] groups, less the query's own position. 0 without a chain.
    pub fn chain_top(&self) -> u32 {
        self.kv_groups
            .iter()
            .filter(|g| g.coverage == Coverage::Chain)
            .filter_map(|g| match g.attention {
                AttentionKind::Sliding { window } => Some(window - 1),
                AttentionKind::Full => None,
            })
            .max()
            .unwrap_or(0)
    }

    /// How far behind a target the chain's bottom layer starts:
    /// `chain_top + (depth - 1) * (window - 1)`. 0 without a chain.
    pub fn chain_reach(&self) -> u32 {
        match self.window_chain {
            Some(c) => self.chain_top() + (c.depth - 1) * (c.window - 1),
            None => 0,
        }
    }

    /// The first position chain layer `layer` (0 = bottom) computes in a row whose top
    /// layer starts at `chain_start` (the row rule of [`crate::executor`]).
    pub fn chain_layer_start(&self, chain_start: u32, layer: u32) -> u32 {
        let c = self.window_chain.expect("a model with a window chain");
        chain_start.saturating_sub((c.depth - 1 - layer) * (c.window - 1))
    }

    /// Logical devices the groups name.
    pub fn num_devices(&self) -> u32 {
        self.kv_groups.iter().map(|g| g.device).max().unwrap_or(0) + 1
    }

    /// Whether the executor keeps a host tier for idle prefix-cache entries.
    pub fn has_host_tier(&self) -> bool {
        self.kv_groups.iter().all(|g| g.host_blocks > 1)
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
            if g.ratio == 0 || !self.block_size.is_multiple_of(g.ratio) {
                return Err(format!(
                    "group {}: ratio {} must divide the block size {}",
                    g.name, g.ratio, self.block_size
                ));
            }
            if g.ratio != 1 && g.attention != AttentionKind::Full {
                return Err(format!("group {}: only full attention pools", g.name));
            }
            if g.coverage == Coverage::Chain && g.attention == AttentionKind::Full {
                return Err(format!(
                    "group {}: a chain-covered group is a sliding window",
                    g.name
                ));
            }
        }
        let chained = |role| {
            self.kv_groups
                .iter()
                .filter(move |g| g.coverage == Coverage::Chain && g.role == role)
        };
        match self.window_chain {
            None => {
                if self.kv_groups.iter().any(|g| g.coverage == Coverage::Chain) {
                    return Err("a chain-covered group needs a window chain".into());
                }
            }
            Some(c) => {
                if c.depth == 0 || c.window == 0 {
                    return Err("the window chain needs a depth and a window".into());
                }
                let mut targets = chained(KvRole::Target);
                let (Some(g), None) = (targets.next(), targets.next()) else {
                    return Err("the window chain is exactly one chain-covered target group".into());
                };
                if g.attention != (AttentionKind::Sliding { window: c.window }) {
                    return Err(format!(
                        "group {} must slide by the chain's window {}",
                        g.name, c.window
                    ));
                }
            }
        }
        let tiered = self.kv_groups.iter().filter(|g| g.host_blocks > 0).count();
        if tiered != 0 && tiered != self.kv_groups.len() {
            return Err("either every KV group has a host tier or none does".into());
        }
        if let Some(g) = self.kv_groups.iter().find(|g| g.host_blocks == 1) {
            return Err(format!(
                "group {} needs at least one allocatable host block",
                g.name
            ));
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
