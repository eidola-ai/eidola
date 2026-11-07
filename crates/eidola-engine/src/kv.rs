//! Paged KV manager: physical block pools per KV group, per-sequence block tables, and the
//! salted prefix cache with idle-TTL, max-age, LRU eviction and zero-on-free.
//!
//! # Memory model
//!
//! Every KV group has its own pool of physical blocks (ids `1..num_blocks`; 0 is the null
//! block). A block is reference counted: each sequence that maps it holds one reference and
//! the prefix cache holds one per entry that records it. When the count reaches zero the
//! block returns to the free list **and a [`Maintenance::Zero`] is queued for it**, so a
//! free block never holds data from a previous owner once the next step's maintenance has
//! run. Blocks on the free list are therefore always zero (or about to be, in the pending
//! maintenance), whether they were freed by a finishing sequence, a sliding window, a
//! speculative rollback, or cache eviction.
//!
//! # Sliding-window groups
//!
//! Sliding-window groups are paged exactly like full-attention groups: a sequence's
//! logical block `i` maps to a physical block in every group. As a sequence advances, a
//! sliding-window block whose positions are all outside the window of the sequence's
//! newest *sealed* block boundary is unmapped and released (it is never read again).
//! Keeping the window behind the sealed boundary rather than the newest position costs at
//! most one extra block per group and means the window for the boundary a later request
//! would resume from is always still present at release.
//!
//! # Prefix cache
//!
//! Full blocks are sealed into the cache under the salted SHA-256 chain of
//! [`crate::hash`]. A cache entry always holds its full-attention blocks. Sliding-window
//! blocks are attached to entries only where they complete the window of a *hit point*,
//! a boundary a later request under the same salt resumes from. A keyed sequence retains
//! the window at three such points:
//!
//! - **branch**: the boundary where its chain left the cached trie at admission (the
//!   deepest boundary whose blocks were cached, if their window was not). A request that
//!   shares a prefix with an earlier one and then diverges finds the earlier one's
//!   full-attention blocks but no window there; it recomputes the prefix, and the window
//!   it computes at the divergence is kept, so every later request that shares the prefix
//!   and diverges anywhere after it hits there;
//! - **regeneration**: the last full block of its prompt before the final token;
//! - **continuation**: its last sealed boundary, where a continuation of the
//!   conversation resumes.
//!
//! The first two are attached as soon as the boundary is sealed (before the window slides
//! past it), so requests that arrive while the sequence is still running can hit them;
//! the third is attached at release. A lookup accepts a hit of `m` blocks only if every
//! full-attention block `0..m` is cached and every sliding-window group has its blocks
//! covering `[first_visible(m * block_size), m * block_size)` attached. Hits are therefore
//! always correct, and windows are retained only where they can be used: at most
//! `ceil((window - 1) / block_size) + 1` blocks per sliding group per point, three points
//! per sequence.
//!
//! Entries form a tree (each chains its parent). Eviction under pressure takes the least
//! recently used unreferenced leaf, then cascades to ancestors that became unreferenced
//! leaves and are not valid hit points themselves. Expiry: an unreferenced entry idle for
//! `idle_ttl_ms`, or any entry older than `max_age_ms`, is removed together with its
//! subtree, both by [`KvManager::sweep`] and lazily by any lookup that walks into it. An
//! in-use entry past its max age is detached (no further hits); its blocks are freed and
//! zeroed when the last sequence using them releases.
//!
//! Sequences admitted with `retain = false` (no client cache key; a fresh random salt)
//! still seal blocks, so a preempted sequence can resume from its own KV, but every entry
//! they created is purged when they finish.
//!
//! # Idle blocks are encrypted
//!
//! A block that a cache entry holds and no sequence maps is *idle*, and is kept encrypted
//! under its lineage's [`KvKey`] ([`crate::kv_cipher`]). Every block is either free (zero,
//! or a zero queued), mapped by at least one sequence (plaintext), or idle (ciphertext),
//! and the host tracks which; a block moves between mapped and idle only through a
//! sequence, which holds its lineage's key:
//!
//! - when the last sequence mapping a held block lets go of it (finish, preemption, or its
//!   window sliding past a retained point), a [`Maintenance::Encrypt`] is queued;
//! - when a sequence maps an idle block (a prefix-cache hit), a [`Maintenance::Decrypt`]
//!   is queued, so concurrent sequences of one lineage share plaintext while any of them
//!   runs;
//! - a block whose entry is evicted or expires while idle is zeroed like any other freed
//!   block, and one whose entry is detached while in use is zeroed when its last user
//!   releases it, never encrypted and kept.
//!
//! Every lineage's blocks are covered alike, whatever the group: full-attention blocks,
//! sliding-window blocks attached at branch, regeneration and continuation points, and the
//! drafter group's blocks with their boundary taps. Requests without a client key get a
//! fresh random key, so even their blocks held across a preemption are ciphertext.
//!
//! Queued encryptions and decryptions are kept per block until the next
//! [`KvManager::take_pending`], so a block is named in at most one of them per step: an
//! encrypt and a decrypt of the same block cancel, and a zero cancels either. Each holds
//! an index into the step's keys, which are copies of the sequences' own and are dropped
//! (zeroed) with the step.
//!
//! # Host tier
//!
//! When the executor keeps a host tier ([`ModelSpec::has_host_tier`]: a second, larger
//! pool per group in the confidential VM's own memory), an idle entry evicted from the
//! device pools under pressure is **demoted** instead of dropped: its blocks are copied,
//! as the ciphertext they already are, into host blocks ([`Maintenance::Demote`]), and its
//! device blocks are freed and zeroed like any other. Each slot of an entry holds its
//! block on the device or on the host, never both. A prefix-cache hit on a demoted block
//! **promotes** it: a fresh device block is allocated, the ciphertext is copied back
//! ([`Maintenance::Promote`]) and the host block is freed; the decryption that mapping
//! the block queues then runs after the copy, so nothing on the host ever holds plaintext
//! or a key. Only the blocks a hit maps are promoted (a sliding group's window, every
//! full-attention block); the rest stay on the host.
//!
//! - **What is demoted**: under device pressure in a group, the least recently used
//!   unreferenced entry holding a device block of that group whose blocks are all
//!   encrypted on the device when the step begins (no encryption or decryption queued, no
//!   promotion landing in them). Any unreferenced entry qualifies, leaf or not: it stays
//!   in the tree, only its bytes move.
//! - **Host pressure**: a host block is taken from the free list, else the least recently
//!   used unreferenced entry holding a host block of that group is dropped (a leaf first,
//!   with the eviction cascade; otherwise with its subtree), its host blocks zeroed. If
//!   the host tier cannot take an entry, the device falls back to dropping one.
//! - **Retention is the same.** Expiry (idle TTL, max age), sweeps and purges remove an
//!   entry wherever its blocks are, and every host block an entry stops holding gets a
//!   [`Maintenance::ZeroHost`]. A host block freed by a promotion stays out of the free
//!   list until the step that zeroes it has been taken (the promotion reads it later in
//!   that step); one freed by a drop may be reused at once by a demotion, which
//!   overwrites every byte of it, and then its zero is cancelled.
//!
//! A step's maintenance is ordered so each of these sees the right bytes: demotions
//! first (they read blocks encrypted by earlier steps), then zeros and slot resets, then
//! promotions (into blocks zeroed above), then encryptions and decryptions (a promoted
//! block's decryption after its copy), then host zeros (after the promotions that read
//! those blocks).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::executor::{Maintenance, Slot, TableUpdate};
use crate::hash::{BlockHash, extend_chain};
use crate::kv_cipher::KvNonce;
use crate::secret::{KvKey, Lineage};
use crate::spec::{AttentionKind, ModelSpec, NULL_BLOCK};

/// Milliseconds on a monotonic clock supplied by the caller.
pub type Millis = u64;

/// Engine-side identifier of a sequence.
pub type SeqId = u64;

/// Prefix-cache lifetime policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachePolicy {
    /// Whether finished sequences' blocks are kept for reuse at all.
    pub enabled: bool,
    /// An unreferenced entry is evicted this long after its last use.
    pub idle_ttl_ms: Millis,
    /// Every entry is evicted (or detached, if in use) this long after it was created.
    pub max_age_ms: Millis,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            idle_ttl_ms: 15 * 60 * 1000,
            max_age_ms: 2 * 60 * 60 * 1000,
        }
    }
}

/// How a sequence leaves the running set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Release {
    /// Finished or cancelled: retain windows if the sequence is keyed, purge its entries if
    /// it is not.
    Finish,
    /// Preempted (or admission rolled back): retain windows so the resume can hit them.
    Preempt,
}

#[derive(Debug)]
struct Pool {
    free: Vec<u32>,
    refs: Vec<u32>,
    /// The cache entry holding each block, or [`NO_ENTRY`]. At most one entry holds a
    /// block (it covers one logical index of one chain).
    holder: Vec<EntryId>,
    /// Whether each block is encrypted once the queued maintenance has run: exactly the
    /// idle blocks (held by an entry, mapped by no sequence).
    encrypted: Vec<bool>,
    /// Blocks a queued promotion writes: their device bytes are not the entry's until the
    /// next step has run, so they cannot be demoted before then.
    landing: Vec<bool>,
}

impl Pool {
    fn new(num_blocks: u32) -> Self {
        // Pop from the end: hand out low ids first (purely cosmetic).
        let free = (1..num_blocks).rev().collect();
        Self {
            free,
            refs: vec![0; num_blocks as usize],
            holder: vec![NO_ENTRY; num_blocks as usize],
            encrypted: vec![false; num_blocks as usize],
            landing: vec![false; num_blocks as usize],
        }
    }

    /// Held by an entry and mapped by no sequence.
    fn idle(&self, b: u32) -> bool {
        let b = b as usize;
        self.holder[b] != NO_ENTRY && self.refs[b] == 1
    }
}

/// A group's host-tier pool. A host block is held by at most one entry and by nothing
/// else (sequences only ever map device blocks), so it needs no reference count.
#[derive(Debug)]
struct HostPool {
    /// Unheld blocks a demotion may write now: zero, or with a zero queued that a
    /// demotion into them cancels.
    free: Vec<u32>,
    /// Unheld blocks a queued promotion still reads: they rejoin `free` when the step
    /// that zeroes them is taken.
    quarantine: Vec<u32>,
    holder: Vec<EntryId>,
}

impl HostPool {
    fn new(host_blocks: u32) -> Self {
        Self {
            free: (1..host_blocks).rev().collect(),
            quarantine: Vec::new(),
            holder: vec![NO_ENTRY; host_blocks as usize],
        }
    }
}

/// Content-free host-tier counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TierStats {
    /// Entries demoted from the device pools to the host tier.
    pub demoted_entries: u64,
    /// Blocks demoted.
    pub demoted_blocks: u64,
    /// Blocks promoted back to the device for a hit.
    pub promoted_blocks: u64,
    /// Entries dropped (with what their removal cascades to) to make room in the host
    /// tier.
    pub host_drops: u64,
    /// Device-pressure evictions that dropped an entry because none could be demoted.
    pub device_drops: u64,
}

type EntryId = u64;

/// [`Pool::holder`] of a block no entry holds (entry ids start at 1).
const NO_ENTRY: EntryId = 0;

/// A queued encryption or decryption of one block.
#[derive(Clone, Copy, Debug)]
struct PendingCrypt {
    encrypt: bool,
    key: u32,
    nonce: KvNonce,
}

/// What [`KvManager::take_pending`] hands to the next step.
#[derive(Debug, Default)]
pub struct PendingWork {
    /// Maintenance, in order: zeros, copies and slot resets as queued, then the
    /// encryptions and decryptions (each naming a block no other operation here names).
    pub maintenance: Vec<Maintenance>,
    /// Table updates, in order.
    pub table_updates: Vec<TableUpdate>,
    /// The keys the encryptions and decryptions name, by index.
    pub kv_keys: Vec<KvKey>,
}

#[derive(Debug)]
struct Entry {
    hash: BlockHash,
    parent: Option<EntryId>,
    depth: u32,
    children: Vec<EntryId>,
    users: u32,
    /// Device block per group, or [`NULL_BLOCK`].
    blocks: Vec<u32>,
    /// Host-tier block per group, or [`NULL_BLOCK`]; never set where `blocks` is.
    host: Vec<u32>,
    created: Millis,
    last_used: Millis,
    lru_key: Option<(Millis, EntryId)>,
    /// The key the entry is indexed under in the host tier's per-group sets, while it is
    /// unreferenced.
    tier_key: Option<Millis>,
}

impl Entry {
    /// Whether the entry holds group `g`'s block, on the device or the host.
    fn holds(&self, g: usize) -> bool {
        self.blocks[g] != NULL_BLOCK || self.host[g] != NULL_BLOCK
    }
}

#[derive(Debug)]
struct SeqKv {
    slot: Slot,
    lineage: Lineage,
    retain: bool,
    hashes: Vec<BlockHash>,
    entries: Vec<EntryId>,
    chain_broken: bool,
    blocks: Vec<Vec<u32>>,
    computed: u32,
    /// Branch and regeneration points not yet sealed, ascending (block boundaries).
    points: Vec<u32>,
    /// Window blocks this sequence attached to entries at those points, as
    /// `(entry, group, block)`, while it runs (so [`KvManager::unpin`] can take them back).
    attached: Vec<(EntryId, usize, u32)>,
    slide_cursor: Vec<u32>,
}

/// The paged KV manager. See the module documentation.
#[derive(Debug)]
pub struct KvManager {
    block_size: u32,
    attention: Vec<AttentionKind>,
    policy: CachePolicy,
    pools: Vec<Pool>,
    entries: HashMap<EntryId, Entry>,
    map: HashMap<BlockHash, EntryId>,
    lru: BTreeSet<(Millis, EntryId)>,
    next_entry: EntryId,
    seqs: HashMap<SeqId, SeqKv>,
    private_entries: HashMap<SeqId, Vec<EntryId>>,
    free_slots: Vec<Slot>,
    pending_maint: Vec<Maintenance>,
    pending_tables: Vec<TableUpdate>,
    pending_crypt: BTreeMap<(u32, u32), PendingCrypt>,
    pending_keys: Vec<KvKey>,
    /// Host-tier pools, one per group; empty when there is no host tier.
    host: Vec<HostPool>,
    /// Per group: unreferenced entries holding a device block of it, by last use.
    dev_lru: Vec<BTreeSet<(Millis, EntryId)>>,
    /// Per group: unreferenced entries holding a host block of it, by last use.
    host_lru: Vec<BTreeSet<(Millis, EntryId)>>,
    pending_demote: Vec<Maintenance>,
    /// Queued promotions: `(group, device block) -> host block`.
    pending_promote: BTreeMap<(u32, u32), u32>,
    pending_host_zero: BTreeSet<(u32, u32)>,
    tier: TierStats,
}

impl KvManager {
    /// A manager for `spec`'s groups and slots.
    pub fn new(spec: &ModelSpec, policy: CachePolicy) -> Self {
        Self {
            block_size: spec.block_size,
            attention: spec.kv_groups.iter().map(|g| g.attention).collect(),
            policy,
            pools: spec
                .kv_groups
                .iter()
                .map(|g| Pool::new(g.num_blocks))
                .collect(),
            entries: HashMap::new(),
            map: HashMap::new(),
            lru: BTreeSet::new(),
            next_entry: 1,
            seqs: HashMap::new(),
            private_entries: HashMap::new(),
            free_slots: (0..spec.num_state_slots).rev().collect(),
            pending_maint: Vec::new(),
            pending_tables: Vec::new(),
            pending_crypt: BTreeMap::new(),
            pending_keys: Vec::new(),
            host: if spec.has_host_tier() && policy.enabled {
                spec.kv_groups
                    .iter()
                    .map(|g| HostPool::new(g.host_blocks))
                    .collect()
            } else {
                Vec::new()
            },
            dev_lru: vec![BTreeSet::new(); spec.kv_groups.len()],
            host_lru: vec![BTreeSet::new(); spec.kv_groups.len()],
            pending_demote: Vec::new(),
            pending_promote: BTreeMap::new(),
            pending_host_zero: BTreeSet::new(),
            tier: TierStats::default(),
        }
    }

    /// Whether idle entries are demoted to a host tier under device pressure.
    pub fn has_host_tier(&self) -> bool {
        !self.host.is_empty()
    }

    /// Host-tier counters.
    pub fn tier_stats(&self) -> TierStats {
        self.tier
    }

    /// The cache policy in force.
    pub fn policy(&self) -> CachePolicy {
        self.policy
    }

    /// Takes the maintenance, table updates and keys accumulated since the last call; they
    /// must be handed to the executor, in order, before or with the next step. Only the
    /// keys a queued operation still names are handed on; the rest are dropped here.
    pub fn take_pending(&mut self) -> PendingWork {
        // Demotions read blocks encrypted by earlier steps, before their zeros; promotions
        // write blocks after their zeros and before their decryptions; host zeros come
        // after the promotions that read those host blocks.
        let mut maintenance = std::mem::take(&mut self.pending_demote);
        maintenance.append(&mut self.pending_maint);
        for ((group, block), host) in std::mem::take(&mut self.pending_promote) {
            self.pools[group as usize].landing[block as usize] = false;
            maintenance.push(Maintenance::Promote { group, host, block });
        }
        let keys = std::mem::take(&mut self.pending_keys);
        let mut used: Vec<Option<u32>> = vec![None; keys.len()];
        let mut kv_keys = Vec::new();
        for ((group, block), p) in std::mem::take(&mut self.pending_crypt) {
            let key = *used[p.key as usize].get_or_insert_with(|| {
                kv_keys.push(keys[p.key as usize].clone());
                kv_keys.len() as u32 - 1
            });
            let nonce = p.nonce;
            maintenance.push(if p.encrypt {
                Maintenance::Encrypt {
                    group,
                    block,
                    key,
                    nonce,
                }
            } else {
                Maintenance::Decrypt {
                    group,
                    block,
                    key,
                    nonce,
                }
            });
        }
        for (group, host) in std::mem::take(&mut self.pending_host_zero) {
            maintenance.push(Maintenance::ZeroHost { group, host });
        }
        for pool in &mut self.host {
            pool.free.append(&mut pool.quarantine);
        }
        PendingWork {
            maintenance,
            table_updates: std::mem::take(&mut self.pending_tables),
            kv_keys,
        }
    }

    /// Whether any maintenance (or a key that only a maintenance step drops) is waiting
    /// for the executor.
    pub fn has_pending_maintenance(&self) -> bool {
        !self.pending_maint.is_empty()
            || !self.pending_crypt.is_empty()
            || !self.pending_keys.is_empty()
            || !self.pending_demote.is_empty()
            || !self.pending_promote.is_empty()
            || !self.pending_host_zero.is_empty()
            || self.host.iter().any(|p| !p.quarantine.is_empty())
    }

    /// Idle-KV keys this manager holds: one per live sequence, and those waiting for the
    /// next step's encryptions and decryptions. Zero once every sequence has released and
    /// the pending work has been taken.
    pub fn kv_keys_held(&self) -> usize {
        self.seqs.len() + self.pending_keys.len()
    }

    /// Free blocks in `group` (not counting evictable cache entries).
    pub fn free_blocks(&self, group: usize) -> usize {
        self.pools[group].free.len()
    }

    /// Whether a state slot is available.
    pub fn has_free_slot(&self) -> bool {
        !self.free_slots.is_empty()
    }

    /// Positions of `id` whose KV is written.
    pub fn computed(&self, id: SeqId) -> u32 {
        self.seqs[&id].computed
    }

    /// The state slot of `id`.
    pub fn slot(&self, id: SeqId) -> Slot {
        self.seqs[&id].slot
    }

    /// Admits a sequence: takes a slot, looks up the longest valid prefix-cache hit for
    /// `tokens` (never covering the last token, which must be recomputed to produce
    /// logits), maps the hit blocks, and returns the number of cached positions. `None` if
    /// no slot is free.
    ///
    /// `prompt_len` is the length of the original prompt (it fixes the regeneration hit
    /// point whose window is retained). A keyed sequence (`retain`) whose chain matches
    /// cached blocks beyond its hit also records that boundary as its branch point.
    ///
    /// Hit blocks that were idle are decrypted (queued) under `lineage`'s key, which the
    /// sequence holds until it is released.
    pub fn admit(
        &mut self,
        id: SeqId,
        lineage: Lineage,
        retain: bool,
        tokens: &[u32],
        prompt_len: u32,
        now: Millis,
    ) -> Option<u32> {
        assert!(!tokens.is_empty(), "a sequence needs at least one token");
        assert!(!self.seqs.contains_key(&id), "sequence admitted twice");
        let slot = self.free_slots.pop()?;
        self.reset_slot(slot);
        let groups = self.pools.len();
        let bs = self.block_size;
        let mut seq = SeqKv {
            slot,
            lineage,
            retain,
            hashes: Vec::new(),
            entries: Vec::new(),
            chain_broken: false,
            blocks: vec![Vec::new(); groups],
            computed: 0,
            points: Vec::new(),
            attached: Vec::new(),
            slide_cursor: vec![0; groups],
        };

        if self.policy.enabled {
            let cap_blocks = (tokens.len() as u32 - 1) / bs;
            let mut hashes = Vec::new();
            extend_chain(
                seq.lineage.salt(),
                &tokens[..(cap_blocks * bs) as usize],
                bs as usize,
                &mut hashes,
            );
            let mut chain = Vec::new();
            for h in &hashes {
                let Some(&eid) = self.map.get(h) else { break };
                if self.expired(eid, now) {
                    self.remove_subtree(eid);
                    break;
                }
                chain.push(eid);
            }
            let branch = chain.len() as u32;
            let mut m = branch;
            while m > 0 && !self.window_covered(&chain, m) {
                m -= 1;
            }
            if self.has_host_tier()
                && let Some(&deepest) = chain.last()
            {
                // Bring back to the device every demoted block the hit maps, shortening
                // the hit while the device cannot take them. The chain is protected
                // meanwhile, so nothing the promotions evict or demote is part of it.
                self.protect(deepest);
                while m > 0 && !self.promote_hit(&chain, m) {
                    m -= 1;
                    while m > 0 && !self.window_covered(&chain, m) {
                        m -= 1;
                    }
                }
                self.unprotect(deepest);
            }
            chain.truncate(m as usize);
            hashes.truncate(m as usize);
            for (idx, &eid) in chain.iter().enumerate() {
                let entry = self.entries.get_mut(&eid).expect("chain entry exists");
                entry.users += 1;
                entry.last_used = now;
                let blocks = entry.blocks.clone();
                self.lru_refresh(eid);
                for (g, &cached) in blocks.iter().enumerate() {
                    let lo = self.attention[g].first_visible(m * bs) / bs;
                    let b = if (idx as u32) >= lo {
                        debug_assert_ne!(cached, NULL_BLOCK, "a hit maps only device blocks");
                        cached
                    } else {
                        NULL_BLOCK
                    };
                    if b != NULL_BLOCK {
                        self.pools[g].refs[b as usize] += 1;
                        self.settle(g, b, Some(seq.lineage.kv_key()));
                        self.pending_tables.push(TableUpdate {
                            slot,
                            group: g as u32,
                            index: idx as u32,
                            block: b,
                        });
                    }
                    seq.blocks[g].push(b);
                }
            }
            seq.entries = chain;
            seq.hashes = hashes;
            seq.computed = m * bs;
            for g in 0..groups {
                seq.slide_cursor[g] = self.attention[g].first_visible(m * bs) / bs;
            }
            // Retention points this sequence will attach its window at once sealed. Only a
            // keyed sequence's entries outlive it, and only sliding groups need them.
            let sliding = self.attention.iter().any(|a| *a != AttentionKind::Full);
            if retain && sliding {
                let regen = (prompt_len.max(1) - 1) / bs;
                let mut points: Vec<u32> = [branch, regen].into_iter().filter(|&p| p > m).collect();
                points.sort_unstable();
                points.dedup();
                seq.points = points;
            }
        }
        let cached = seq.computed;
        self.seqs.insert(id, seq);
        Some(cached)
    }

    /// Ensures blocks are mapped for positions `[0, upto)` in every group, evicting cache
    /// entries as needed. On failure nothing is allocated and `false` is returned.
    pub fn allocate(&mut self, id: SeqId, upto: u32) -> bool {
        let needed = upto.div_ceil(self.block_size);
        let have = self.seqs[&id].blocks[0].len() as u32;
        if needed <= have {
            return true;
        }
        let groups = self.pools.len();
        let mut got: Vec<Vec<u32>> = vec![Vec::new(); groups];
        let count = (needed - have) as usize;
        for (g, got_g) in got.iter_mut().enumerate() {
            for _ in 0..count {
                match self.alloc_block(g) {
                    Some(b) => got_g.push(b),
                    None => {
                        // Untouched blocks go straight back: they are still zero.
                        for (g2, list) in got.iter().enumerate() {
                            for &b in list {
                                self.pools[g2].refs[b as usize] = 0;
                                self.pools[g2].free.push(b);
                            }
                        }
                        return false;
                    }
                }
            }
        }
        let seq = self.seqs.get_mut(&id).expect("live sequence");
        for (g, list) in got.into_iter().enumerate() {
            for (k, b) in list.into_iter().enumerate() {
                let index = have + k as u32;
                seq.blocks[g].push(b);
                self.pending_tables.push(TableUpdate {
                    slot: seq.slot,
                    group: g as u32,
                    index,
                    block: b,
                });
            }
        }
        true
    }

    /// Gives back blocks [`KvManager::allocate`] mapped for `id` past position `upto`
    /// during the step being planned (a draft reservation the scheduler revokes). Those
    /// blocks were taken from the free list this step and nothing has written them, so
    /// they return to it without a zero; their table entries are reset to the null block.
    /// `upto` may not cut into positions with valid KV.
    pub fn shrink(&mut self, id: SeqId, upto: u32) {
        let mut seq = self.seqs.remove(&id).expect("live sequence");
        assert!(upto >= seq.computed, "shrink into computed KV");
        let keep = upto.div_ceil(self.block_size) as usize;
        for g in 0..self.pools.len() {
            while seq.blocks[g].len() > keep {
                let idx = seq.blocks[g].len() - 1;
                let b = seq.blocks[g].pop().expect("non-empty");
                if b == NULL_BLOCK {
                    continue;
                }
                let r = &mut self.pools[g].refs[b as usize];
                assert_eq!(*r, 1, "a reserved block is owned by its sequence alone");
                *r = 0;
                self.pools[g].free.push(b);
                self.pending_tables.push(TableUpdate {
                    slot: seq.slot,
                    group: g as u32,
                    index: idx as u32,
                    block: NULL_BLOCK,
                });
            }
        }
        self.seqs.insert(id, seq);
    }

    /// Gives up `id`'s branch and regeneration retention: forgets the points not yet
    /// sealed and takes back the window blocks it already attached to cache entries at the
    /// others, freeing those no sequence maps any more (the ones already behind its own
    /// window). The sequence itself loses nothing; those points are simply not retained.
    /// Returns whether any block was freed.
    pub fn unpin(&mut self, id: SeqId) -> bool {
        let seq = self.seqs.get_mut(&id).expect("live sequence");
        seq.points.clear();
        let attached = std::mem::take(&mut seq.attached);
        let mut freed = false;
        for (eid, g, b) in attached {
            let Some(entry) = self.entries.get_mut(&eid) else {
                continue;
            };
            if entry.blocks[g] != b {
                continue;
            }
            entry.blocks[g] = NULL_BLOCK;
            self.pools[g].holder[b as usize] = NO_ENTRY;
            freed |= self.pools[g].refs[b as usize] == 1;
            self.decref(g, b);
            self.settle(g, b, None);
        }
        freed
    }

    /// Records that positions `[0, computed)` of `id` hold valid KV after a step: releases
    /// blocks past it (speculative rollback), seals newly full blocks into the cache, and
    /// releases sliding-window blocks that left the window.
    pub fn commit(&mut self, id: SeqId, computed: u32, tokens: &[u32], now: Millis) {
        let bs = self.block_size;
        let groups = self.pools.len();
        let mut seq = self.seqs.remove(&id).expect("live sequence");
        assert!(computed as usize <= tokens.len());
        assert!(
            computed.div_ceil(bs) as usize <= seq.blocks[0].len(),
            "commit past allocation"
        );
        seq.computed = computed;

        // Rollback: blocks that start at or after `computed` hold only rejected positions.
        let keep = computed.div_ceil(bs) as usize;
        for g in 0..groups {
            while seq.blocks[g].len() > keep {
                let idx = seq.blocks[g].len() - 1;
                self.unmap(&mut seq, g, idx);
                seq.blocks[g].pop();
            }
        }

        // Seal full blocks. Every group's KV for a block depends only on the tokens the
        // block covers (the executor contract), so a full block is final in every group.
        let sealable = computed / bs;
        if self.policy.enabled && !seq.chain_broken {
            extend_chain(
                seq.lineage.salt(),
                &tokens[..(sealable * bs) as usize],
                bs as usize,
                &mut seq.hashes,
            );
            while (seq.entries.len() as u32) < sealable {
                if !self.seal_next(&mut seq, id, now) {
                    seq.chain_broken = true;
                    break;
                }
            }
        }

        // Attach the windows of retention points sealed by now, before they slide away:
        // the window of a boundary at or past the previous sealed one is still mapped.
        while let Some(&point) = seq.points.first() {
            if point > sealable && !seq.chain_broken {
                break;
            }
            seq.points.remove(0);
            if point <= seq.entries.len() as u32 {
                let attached = self.attach_window(&seq, point);
                seq.attached.extend(attached);
            }
        }

        // Slide windows behind the newest sealed boundary.
        for g in 0..groups {
            if let AttentionKind::Sliding { .. } = self.attention[g] {
                let lo_keep = self.attention[g].first_visible(sealable * bs) / bs;
                while seq.slide_cursor[g] < lo_keep {
                    let idx = seq.slide_cursor[g];
                    seq.slide_cursor[g] += 1;
                    if (idx as usize) < seq.blocks[g].len() {
                        self.unmap(&mut seq, g, idx as usize);
                    }
                }
            }
        }
        self.seqs.insert(id, seq);
    }

    /// Releases a sequence's slot and blocks. See [`Release`]. Blocks the cache keeps and no
    /// other sequence maps are encrypted (queued) under the sequence's key. The sequence's
    /// lineage is dropped here; the copy of its key those encryptions carry goes with the
    /// next step.
    pub fn release(&mut self, id: SeqId, kind: Release, now: Millis) {
        let groups = self.pools.len();
        let seq = self.seqs.remove(&id).expect("live sequence");
        let retain = kind == Release::Preempt || seq.retain;

        if self.policy.enabled && retain {
            // The continuation point; branch and regeneration points were attached when
            // they were sealed.
            self.attach_window(&seq, seq.entries.len() as u32);
        }

        for g in 0..groups {
            for idx in 0..seq.blocks[g].len() {
                let b = seq.blocks[g][idx];
                if b != NULL_BLOCK {
                    self.decref(g, b);
                    self.settle(g, b, Some(seq.lineage.kv_key()));
                }
            }
        }
        for &eid in &seq.entries {
            if let Some(entry) = self.entries.get_mut(&eid) {
                entry.users -= 1;
                entry.last_used = now;
                self.lru_refresh(eid);
            }
        }
        self.reset_slot(seq.slot);
        self.free_slots.push(seq.slot);

        if kind == Release::Finish {
            self.forget(id);
        }
    }

    /// Purges the entries a private (unkeyed) sequence sealed; called when it finishes, or
    /// when it is cancelled while waiting after a preemption.
    pub fn forget(&mut self, id: SeqId) {
        if let Some(list) = self.private_entries.remove(&id) {
            // Leaves first: entries were created in chain order.
            for &eid in list.iter().rev() {
                let removable = self
                    .entries
                    .get(&eid)
                    .is_some_and(|e| e.users == 0 && e.children.is_empty());
                if removable {
                    self.remove_entry(eid);
                }
            }
        }
    }

    /// Evicts every expired entry (with its subtree) and returns how many were removed.
    /// Zero commands for the freed blocks are queued.
    pub fn sweep(&mut self, now: Millis) -> usize {
        let mut expired: Vec<(u32, EntryId)> = self
            .entries
            .iter()
            .filter(|(id, _)| self.expired(**id, now))
            .map(|(id, e)| (e.depth, *id))
            .collect();
        expired.sort_unstable();
        let before = self.entries.len();
        for (_, eid) in expired {
            if self.entries.contains_key(&eid) {
                self.remove_subtree(eid);
            }
        }
        before - self.entries.len()
    }

    /// Number of live cache entries.
    pub fn cache_entries(&self) -> usize {
        self.entries.len()
    }

    /// Physical blocks referenced by cache entries, as `(group, block)` pairs (testing).
    pub fn cache_blocks(&self) -> BTreeSet<(u32, u32)> {
        let mut out = BTreeSet::new();
        for e in self.entries.values() {
            for (g, &b) in e.blocks.iter().enumerate() {
                if b != NULL_BLOCK {
                    out.insert((g as u32, b));
                }
            }
        }
        out
    }

    /// Physical blocks mapped by live sequences, as `(group, block)` pairs (testing).
    pub fn sequence_blocks(&self) -> BTreeSet<(u32, u32)> {
        let mut out = BTreeSet::new();
        for s in self.seqs.values() {
            for (g, list) in s.blocks.iter().enumerate() {
                for &b in list {
                    if b != NULL_BLOCK {
                        out.insert((g as u32, b));
                    }
                }
            }
        }
        out
    }

    /// Host-tier blocks held by cache entries, as `(group, host block)` pairs (testing).
    pub fn host_cache_blocks(&self) -> BTreeSet<(u32, u32)> {
        let mut out = BTreeSet::new();
        for e in self.entries.values() {
            for (g, &h) in e.host.iter().enumerate() {
                if h != NULL_BLOCK {
                    out.insert((g as u32, h));
                }
            }
        }
        out
    }

    /// Host-tier blocks no entry holds (free or waiting for their zero), as `(group, host
    /// block)` pairs (testing).
    pub fn free_host_set(&self) -> BTreeSet<(u32, u32)> {
        let mut out = BTreeSet::new();
        for (g, p) in self.host.iter().enumerate() {
            for &h in p.free.iter().chain(&p.quarantine) {
                out.insert((g as u32, h));
            }
        }
        out
    }

    /// Host blocks with a queued, not yet executed zero (testing).
    pub fn pending_host_zeros(&self) -> BTreeSet<(u32, u32)> {
        self.pending_host_zero.clone()
    }

    /// Queued, not yet executed demotions and promotions as `(group, block, host,
    /// demote)` (testing).
    pub fn pending_transfers(&self) -> Vec<(u32, u32, u32, bool)> {
        let mut out: Vec<_> = self
            .pending_demote
            .iter()
            .filter_map(|m| match *m {
                Maintenance::Demote { group, block, host } => Some((group, block, host, true)),
                _ => None,
            })
            .collect();
        out.extend(
            self.pending_promote
                .iter()
                .map(|(&(g, b), &h)| (g, b, h, false)),
        );
        out
    }

    /// Blocks held by a cache entry and mapped by no sequence, as `(group, block)` pairs:
    /// the ones held encrypted (testing).
    pub fn idle_blocks(&self) -> BTreeSet<(u32, u32)> {
        let mut out = BTreeSet::new();
        for (g, p) in self.pools.iter().enumerate() {
            for b in 1..p.refs.len() as u32 {
                if p.idle(b) {
                    out.insert((g as u32, b));
                }
            }
        }
        out
    }

    /// Queued, not yet executed encryptions (`true`) and decryptions (`false`) by
    /// `(group, block)` (testing).
    pub fn pending_crypts(&self) -> BTreeMap<(u32, u32), bool> {
        self.pending_crypt
            .iter()
            .map(|(&k, p)| (k, p.encrypt))
            .collect()
    }

    /// Blocks with a queued, not yet executed zero, as `(group, block)` pairs (testing).
    pub fn pending_zeros(&self) -> BTreeSet<(u32, u32)> {
        self.pending_maint
            .iter()
            .filter_map(|m| match *m {
                Maintenance::Zero { group, block } => Some((group, block)),
                _ => None,
            })
            .collect()
    }

    /// Slots with a queued, not yet executed reset (testing).
    pub fn pending_slot_resets(&self) -> BTreeSet<Slot> {
        self.pending_maint
            .iter()
            .filter_map(|m| match *m {
                Maintenance::ResetSlot { slot } => Some(slot),
                _ => None,
            })
            .collect()
    }

    /// Slots no sequence holds (testing).
    pub fn free_slot_set(&self) -> BTreeSet<Slot> {
        self.free_slots.iter().copied().collect()
    }

    /// Free-list blocks as `(group, block)` pairs (testing).
    pub fn free_block_set(&self) -> BTreeSet<(u32, u32)> {
        let mut out = BTreeSet::new();
        for (g, p) in self.pools.iter().enumerate() {
            for &b in &p.free {
                out.insert((g as u32, b));
            }
        }
        out
    }

    /// Between steps (after commit, before the next allocation), no sequence holds blocks
    /// beyond the positions with valid KV: rejected speculative positions were rolled back.
    pub fn check_no_reservations(&self) {
        for s in self.seqs.values() {
            let want = s.computed.div_ceil(self.block_size) as usize;
            for list in &s.blocks {
                assert!(
                    list.len() <= want,
                    "sequence holds blocks past its computed KV"
                );
            }
        }
    }

    /// Verifies every structural invariant; panics with a description on violation.
    pub fn check_invariants(&self) {
        let groups = self.pools.len();
        for g in 0..groups {
            let pool = &self.pools[g];
            let mut expected = vec![0u32; pool.refs.len()];
            for s in self.seqs.values() {
                for &b in &s.blocks[g] {
                    if b != NULL_BLOCK {
                        expected[b as usize] += 1;
                    }
                }
            }
            for e in self.entries.values() {
                if e.blocks[g] != NULL_BLOCK {
                    expected[e.blocks[g] as usize] += 1;
                }
            }
            assert_eq!(
                expected, pool.refs,
                "group {g}: refcounts disagree with owners"
            );
            assert_eq!(pool.refs[NULL_BLOCK as usize], 0, "null block referenced");
            let mut seen = vec![false; pool.refs.len()];
            for &b in &pool.free {
                assert_ne!(b, NULL_BLOCK, "null block on free list");
                assert!(!seen[b as usize], "group {g}: block {b} double-freed");
                seen[b as usize] = true;
                assert_eq!(
                    pool.refs[b as usize], 0,
                    "group {g}: referenced block {b} is free"
                );
            }
            for (b, (&free, &refs)) in seen.iter().zip(&pool.refs).enumerate().skip(1) {
                assert!(
                    free || refs > 0,
                    "group {g}: block {b} leaked (unreferenced, not free)"
                );
            }
            let mut holders = vec![NO_ENTRY; pool.refs.len()];
            for (&eid, e) in &self.entries {
                if e.blocks[g] != NULL_BLOCK {
                    holders[e.blocks[g] as usize] = eid;
                }
            }
            assert_eq!(holders, pool.holder, "group {g}: block holders disagree");
            for b in 1..pool.refs.len() as u32 {
                assert_eq!(
                    pool.encrypted[b as usize],
                    pool.idle(b),
                    "group {g}: block {b} is not encrypted exactly while idle"
                );
            }
        }
        for (&(g, b), p) in &self.pending_crypt {
            assert_eq!(
                self.pools[g as usize].encrypted[b as usize], p.encrypt,
                "group {g}: a queued crypt disagrees with block {b}'s state"
            );
            assert!((p.key as usize) < self.pending_keys.len(), "dangling key");
        }
        self.check_host_tier();
        let mut users: HashMap<EntryId, u32> = HashMap::new();
        for s in self.seqs.values() {
            for (i, eid) in s.entries.iter().enumerate() {
                if let Some(e) = self.entries.get(eid) {
                    *users.entry(*eid).or_default() += 1;
                    assert_eq!(e.depth as usize, i, "sequence chain depth mismatch");
                }
            }
        }
        for (id, e) in &self.entries {
            assert_eq!(
                users.get(id).copied().unwrap_or(0),
                e.users,
                "entry users miscounted"
            );
            assert_eq!(
                self.map.get(&e.hash),
                Some(id),
                "entry missing from hash map"
            );
            if let Some(p) = e.parent {
                let parent = self.entries.get(&p).expect("entry parent exists");
                assert!(parent.children.contains(id), "parent does not list child");
                assert_eq!(parent.depth + 1, e.depth);
            } else {
                assert_eq!(e.depth, 0);
            }
            for c in &e.children {
                assert_eq!(self.entries[c].parent, Some(*id));
            }
            for g in 0..e.blocks.len() {
                if self.attention[g] == AttentionKind::Full {
                    assert!(e.holds(g), "entry lacks a full-attention block");
                }
            }
            let evictable = e.users == 0 && e.children.is_empty();
            assert_eq!(evictable, e.lru_key.is_some(), "lru membership wrong");
            if let Some(k) = e.lru_key {
                assert!(self.lru.contains(&k));
            }
        }
        assert_eq!(self.map.len(), self.entries.len());
        assert_eq!(
            self.lru.len(),
            self.entries
                .values()
                .filter(|e| e.lru_key.is_some())
                .count()
        );
        let used_slots: Vec<Slot> = self.seqs.values().map(|s| s.slot).collect();
        for s in &used_slots {
            assert!(!self.free_slots.contains(s), "slot both used and free");
        }
    }

    fn check_host_tier(&self) {
        for (g, pool) in self.pools.iter().enumerate() {
            for b in 1..pool.landing.len() as u32 {
                assert_eq!(
                    pool.landing[b as usize],
                    self.pending_promote.contains_key(&(g as u32, b)),
                    "group {g}: block {b} landing disagrees with the queued promotions"
                );
            }
        }
        for &(g, b) in self.pending_promote.keys() {
            let pool = &self.pools[g as usize];
            assert_ne!(
                pool.holder[b as usize], NO_ENTRY,
                "group {g}: a promotion lands in block {b}, which no entry holds"
            );
        }
        if self.host.is_empty() {
            assert!(
                self.entries
                    .values()
                    .all(|e| e.host.iter().all(|&h| h == NULL_BLOCK)),
                "a host block is held without a host tier"
            );
            return;
        }
        for (g, pool) in self.host.iter().enumerate() {
            let n = pool.holder.len();
            let mut holders = vec![NO_ENTRY; n];
            for (&eid, e) in &self.entries {
                assert!(
                    e.blocks[g] == NULL_BLOCK || e.host[g] == NULL_BLOCK,
                    "group {g}: entry {eid} holds a block in both tiers"
                );
                if e.host[g] != NULL_BLOCK {
                    assert_eq!(
                        holders[e.host[g] as usize], NO_ENTRY,
                        "group {g}: host block {} held twice",
                        e.host[g]
                    );
                    holders[e.host[g] as usize] = eid;
                }
            }
            assert_eq!(holders, pool.holder, "group {g}: host holders disagree");
            let mut seen = vec![false; n];
            for &h in pool.free.iter().chain(&pool.quarantine) {
                assert_ne!(h, NULL_BLOCK, "null host block on a free list");
                assert!(!seen[h as usize], "group {g}: host block {h} freed twice");
                seen[h as usize] = true;
                assert_eq!(
                    pool.holder[h as usize], NO_ENTRY,
                    "group {g}: held host block {h} is free"
                );
            }
            for (h, (&free, &holder)) in seen.iter().zip(&pool.holder).enumerate().skip(1) {
                assert!(
                    free || holder != NO_ENTRY,
                    "group {g}: host block {h} leaked"
                );
            }
            for &h in &pool.quarantine {
                assert!(
                    self.pending_host_zero.contains(&(g as u32, h)),
                    "group {g}: quarantined host block {h} has no zero queued"
                );
            }
            let (mut dev, mut host) = (BTreeSet::new(), BTreeSet::new());
            for (&eid, e) in &self.entries {
                if e.users == 0 {
                    let k = e.tier_key.expect("an unreferenced entry is indexed");
                    if e.blocks[g] != NULL_BLOCK {
                        dev.insert((k, eid));
                    }
                    if e.host[g] != NULL_BLOCK {
                        host.insert((k, eid));
                    }
                }
            }
            assert_eq!(dev, self.dev_lru[g], "group {g}: device tier index wrong");
            assert_eq!(host, self.host_lru[g], "group {g}: host tier index wrong");
        }
    }

    // ----- internals -----

    fn reset_slot(&mut self, slot: Slot) {
        // Reset precedes every table update in execution order, so pending updates for this
        // slot would be wiped anyway; dropping them keeps a reused slot clean.
        self.pending_tables.retain(|u| u.slot != slot);
        self.pending_maint.push(Maintenance::ResetSlot { slot });
    }

    fn unmap(&mut self, seq: &mut SeqKv, g: usize, idx: usize) {
        let b = seq.blocks[g][idx];
        if b == NULL_BLOCK {
            return;
        }
        seq.blocks[g][idx] = NULL_BLOCK;
        self.pending_tables.push(TableUpdate {
            slot: seq.slot,
            group: g as u32,
            index: idx as u32,
            block: NULL_BLOCK,
        });
        self.decref(g, b);
        self.settle(g, b, Some(seq.lineage.kv_key()));
    }

    fn decref(&mut self, g: usize, b: u32) {
        let pool = &mut self.pools[g];
        let r = &mut pool.refs[b as usize];
        assert!(*r > 0, "group {g}: decref of free block {b}");
        *r -= 1;
        if *r == 0 {
            assert_eq!(
                pool.holder[b as usize], NO_ENTRY,
                "group {g}: block {b} freed while an entry holds it"
            );
            pool.free.push(b);
            // A zeroed block holds nothing to encrypt or decrypt: drop any queued crypt,
            // and any promotion into it (the zero runs first, so it would land in a free
            // block).
            pool.encrypted[b as usize] = false;
            if std::mem::take(&mut pool.landing[b as usize]) {
                self.pending_promote.remove(&(g as u32, b));
            }
            self.pending_crypt.remove(&(g as u32, b));
            self.pending_maint.push(Maintenance::Zero {
                group: g as u32,
                block: b,
            });
        }
    }

    /// Brings block `b`'s encryption in line with its ownership after a reference changed:
    /// encrypted exactly while idle. A block becomes idle or mapped again only through a
    /// sequence, whose lineage key `key` is; changes on the entry side (an entry dropping
    /// a block a sequence still maps, or a free) never need one.
    fn settle(&mut self, g: usize, b: u32, key: Option<&KvKey>) {
        let pool = &self.pools[g];
        let want = pool.idle(b);
        if pool.encrypted[b as usize] == want {
            return;
        }
        let key = key.expect("only a sequence moves a block between idle and mapped");
        self.pools[g].encrypted[b as usize] = want;
        if let Some(p) = self.pending_crypt.remove(&(g as u32, b)) {
            // The opposite operation was queued and has not run: the device is already
            // in the wanted state.
            debug_assert_ne!(p.encrypt, want);
            return;
        }
        // The nonce comes from the holding entry now, while one holds the block (a decrypt
        // may run after the entry was detached).
        let holder = self.pools[g].holder[b as usize];
        assert_ne!(
            holder, NO_ENTRY,
            "group {g}: block {b} changes encryption with no entry holding it"
        );
        let nonce = self.entries[&holder].hash.kv_nonce(g as u32);
        let key = match self.pending_keys.iter().position(|k| k.ct_eq(key)) {
            Some(i) => i as u32,
            None => {
                self.pending_keys.push(key.clone());
                self.pending_keys.len() as u32 - 1
            }
        };
        self.pending_crypt.insert(
            (g as u32, b),
            PendingCrypt {
                encrypt: want,
                key,
                nonce,
            },
        );
    }

    fn alloc_block(&mut self, g: usize) -> Option<u32> {
        loop {
            if let Some(b) = self.pools[g].free.pop() {
                self.pools[g].refs[b as usize] = 1;
                return Some(b);
            }
            let reclaimed = if self.has_host_tier() {
                self.reclaim(g)
            } else {
                self.evict_one()
            };
            if !reclaimed {
                return None;
            }
        }
    }

    fn expired(&self, eid: EntryId, now: Millis) -> bool {
        let e = &self.entries[&eid];
        now.saturating_sub(e.created) >= self.policy.max_age_ms
            || (e.users == 0 && now.saturating_sub(e.last_used) >= self.policy.idle_ttl_ms)
    }

    fn lru_refresh(&mut self, eid: EntryId) {
        let e = self.entries.get_mut(&eid).expect("entry");
        if let Some(k) = e.lru_key.take() {
            self.lru.remove(&k);
        }
        if e.users == 0 && e.children.is_empty() {
            let k = (e.last_used, eid);
            e.lru_key = Some(k);
            self.lru.insert(k);
        }
        if self.host.is_empty() {
            return;
        }
        if let Some(k) = e.tier_key.take() {
            for g in 0..self.dev_lru.len() {
                self.dev_lru[g].remove(&(k, eid));
                self.host_lru[g].remove(&(k, eid));
            }
        }
        if e.users == 0 {
            let k = e.last_used;
            e.tier_key = Some(k);
            for g in 0..self.dev_lru.len() {
                if e.blocks[g] != NULL_BLOCK {
                    self.dev_lru[g].insert((k, eid));
                }
                if e.host[g] != NULL_BLOCK {
                    self.host_lru[g].insert((k, eid));
                }
            }
        }
    }

    /// Whether the window of every sliding group is attached for a hit of `m` blocks along
    /// `chain`.
    fn window_covered(&self, chain: &[EntryId], m: u32) -> bool {
        let bs = self.block_size;
        for (g, att) in self.attention.iter().enumerate() {
            if *att == AttentionKind::Full {
                continue;
            }
            let lo = att.first_visible(m * bs) / bs;
            for &eid in &chain[lo as usize..m as usize] {
                if !self.entries[&eid].holds(g) {
                    return false;
                }
            }
        }
        true
    }

    /// Attaches `seq`'s sliding-window blocks covering the window before boundary `point`
    /// to the entries sealed there, where an entry has none yet. Returns what it attached.
    fn attach_window(&mut self, seq: &SeqKv, point: u32) -> Vec<(EntryId, usize, u32)> {
        let mut attached = Vec::new();
        if point == 0 || point > seq.entries.len() as u32 {
            return attached;
        }
        let bs = self.block_size;
        for g in 0..self.pools.len() {
            if self.attention[g] == AttentionKind::Full {
                continue;
            }
            let lo = self.attention[g].first_visible(point * bs) / bs;
            for idx in lo..point {
                let eid = seq.entries[idx as usize];
                let Some(b) = seq.blocks[g].get(idx as usize).copied() else {
                    continue;
                };
                if b == NULL_BLOCK {
                    continue;
                }
                if let Some(entry) = self.entries.get_mut(&eid)
                    && !entry.holds(g)
                {
                    entry.blocks[g] = b;
                    let pool = &mut self.pools[g];
                    debug_assert_eq!(pool.holder[b as usize], NO_ENTRY);
                    pool.holder[b as usize] = eid;
                    pool.refs[b as usize] += 1;
                    attached.push((eid, g, b));
                }
            }
        }
        attached
    }

    fn is_hit_point(&self, eid: EntryId) -> bool {
        let mut chain = Vec::new();
        let mut cur = Some(eid);
        while let Some(id) = cur {
            chain.push(id);
            cur = self.entries[&id].parent;
        }
        chain.reverse();
        let m = chain.len() as u32;
        self.window_covered(&chain, m)
    }

    fn evict_one(&mut self) -> bool {
        let Some(&(_, eid)) = self.lru.iter().next() else {
            return false;
        };
        self.drop_leaf(eid);
        true
    }

    /// Removes an unreferenced leaf, then the ancestors that became unreferenced leaves
    /// and are not valid hit points themselves.
    fn drop_leaf(&mut self, eid: EntryId) {
        let mut parent = self.entries[&eid].parent;
        self.remove_entry(eid);
        while let Some(p) = parent {
            let e = &self.entries[&p];
            if e.users == 0 && e.children.is_empty() && !self.is_hit_point(p) {
                parent = e.parent;
                self.remove_entry(p);
            } else {
                break;
            }
        }
    }

    /// Drops the least recently used unreferenced entry holding a block of group `g` on
    /// the host (`host`) or the device: a leaf if there is one, with the eviction cascade;
    /// otherwise the first one, with its subtree, whose every entry is unreferenced too.
    /// Returns false if there is none.
    fn drop_from(&mut self, g: usize, host: bool) -> bool {
        let set = if host {
            &self.host_lru[g]
        } else {
            &self.dev_lru[g]
        };
        let Some(&(_, first)) = set.iter().next() else {
            return false;
        };
        let leaf = set
            .iter()
            .map(|&(_, e)| e)
            .find(|e| self.entries[e].children.is_empty());
        match leaf {
            Some(eid) => self.drop_leaf(eid),
            None => self.remove_subtree(first),
        }
        true
    }

    /// Makes room in device group `g` with a host tier: demotes the least recently used
    /// demotable entry holding a block of `g`, or, if none can be demoted, drops one.
    fn reclaim(&mut self, g: usize) -> bool {
        let candidate = self.dev_lru[g]
            .iter()
            .map(|&(_, e)| e)
            .find(|&e| self.settled(e));
        if let Some(eid) = candidate
            && self.demote(eid)
        {
            return true;
        }
        let dropped = self.drop_from(g, false);
        if dropped {
            self.tier.device_drops += 1;
        }
        dropped
    }

    /// Whether every device block of an unreferenced entry is ciphertext on the device
    /// as the next step begins: encrypted, with no encryption, decryption or promotion
    /// queued for it.
    fn settled(&self, eid: EntryId) -> bool {
        let e = &self.entries[&eid];
        e.blocks.iter().enumerate().all(|(g, &b)| {
            let pool = &self.pools[g];
            b == NULL_BLOCK
                || (pool.encrypted[b as usize]
                    && !pool.landing[b as usize]
                    && !self.pending_crypt.contains_key(&(g as u32, b)))
        })
    }

    /// Counts a transient user on `eid` and every ancestor, so nothing an eviction,
    /// demotion or host drop does meanwhile removes them.
    fn protect(&mut self, eid: EntryId) {
        let mut cur = Some(eid);
        while let Some(id) = cur {
            let e = self.entries.get_mut(&id).expect("protected entry exists");
            e.users += 1;
            cur = e.parent;
            self.lru_refresh(id);
        }
    }

    /// Undoes [`KvManager::protect`].
    fn unprotect(&mut self, eid: EntryId) {
        let mut cur = Some(eid);
        while let Some(id) = cur {
            let e = self.entries.get_mut(&id).expect("protected entry exists");
            e.users -= 1;
            cur = e.parent;
            self.lru_refresh(id);
        }
    }

    /// Moves an unreferenced, settled entry's device blocks to the host tier: queues each
    /// block's copy and frees (zeroes) the device block. Returns false, changing nothing,
    /// if the host tier cannot take it.
    fn demote(&mut self, eid: EntryId) -> bool {
        self.protect(eid);
        let groups: Vec<usize> = (0..self.pools.len())
            .filter(|&g| self.entries[&eid].blocks[g] != NULL_BLOCK)
            .collect();
        let mut got = Vec::with_capacity(groups.len());
        for g in groups {
            match self.alloc_host(g) {
                Some(h) => got.push((g, h)),
                None => {
                    // Nothing has written them: they go back as they were (a cancelled
                    // zero included, which the next demotion into them cancels again).
                    for (g, h) in got {
                        self.host[g].free.push(h);
                        self.pending_host_zero.insert((g as u32, h));
                    }
                    self.unprotect(eid);
                    return false;
                }
            }
        }
        for (g, h) in got {
            let e = self.entries.get_mut(&eid).expect("demoted entry exists");
            let d = std::mem::replace(&mut e.blocks[g], NULL_BLOCK);
            e.host[g] = h;
            self.host[g].holder[h as usize] = eid;
            self.pools[g].holder[d as usize] = NO_ENTRY;
            self.pending_demote.push(Maintenance::Demote {
                group: g as u32,
                block: d,
                host: h,
            });
            self.decref(g, d);
            self.tier.demoted_blocks += 1;
        }
        self.tier.demoted_entries += 1;
        self.unprotect(eid);
        true
    }

    /// A free host block of group `g`, dropping host-tier entries if there is none.
    fn alloc_host(&mut self, g: usize) -> Option<u32> {
        loop {
            if let Some(h) = self.host[g].free.pop() {
                // The demotion that takes it overwrites every byte.
                self.pending_host_zero.remove(&(g as u32, h));
                return Some(h);
            }
            if !self.drop_from(g, true) {
                return None;
            }
            self.tier.host_drops += 1;
        }
    }

    /// Releases host block `h` of group `g`: queues its zero, and returns it to the free
    /// list, or, when a queued promotion still reads it (`read_later`), to the quarantine.
    fn free_host(&mut self, g: usize, h: u32, read_later: bool) {
        let pool = &mut self.host[g];
        pool.holder[h as usize] = NO_ENTRY;
        if read_later {
            pool.quarantine.push(h);
        } else {
            pool.free.push(h);
        }
        self.pending_host_zero.insert((g as u32, h));
    }

    /// Promotes every demoted block a hit of `m` blocks along `chain` maps (every
    /// full-attention block, a sliding group's window). Returns false when the device
    /// cannot take one; what was promoted stays promoted.
    fn promote_hit(&mut self, chain: &[EntryId], m: u32) -> bool {
        let bs = self.block_size;
        for (idx, &eid) in chain[..m as usize].iter().enumerate() {
            for g in 0..self.pools.len() {
                let lo = self.attention[g].first_visible(m * bs) / bs;
                if (idx as u32) >= lo
                    && self.entries[&eid].host[g] != NULL_BLOCK
                    && !self.promote(eid, g)
                {
                    return false;
                }
            }
        }
        true
    }

    /// Moves group `g`'s block of a protected entry from the host tier to a fresh device
    /// block. The device block holds the entry's ciphertext once the queued copy has run;
    /// mapping it then queues its decryption.
    fn promote(&mut self, eid: EntryId, g: usize) -> bool {
        let Some(d) = self.alloc_block(g) else {
            return false;
        };
        let e = self.entries.get_mut(&eid).expect("promoted entry exists");
        let h = std::mem::replace(&mut e.host[g], NULL_BLOCK);
        e.blocks[g] = d;
        let pool = &mut self.pools[g];
        pool.holder[d as usize] = eid;
        pool.encrypted[d as usize] = true;
        pool.landing[d as usize] = true;
        self.pending_promote.insert((g as u32, d), h);
        self.free_host(g, h, true);
        self.tier.promoted_blocks += 1;
        true
    }

    fn remove_subtree(&mut self, eid: EntryId) {
        let children = self.entries[&eid].children.clone();
        for c in children {
            self.remove_subtree(c);
        }
        self.remove_entry(eid);
    }

    fn remove_entry(&mut self, eid: EntryId) {
        let e = self.entries.remove(&eid).expect("entry");
        debug_assert!(e.children.is_empty());
        if let Some(k) = e.lru_key {
            self.lru.remove(&k);
        }
        if let Some(k) = e.tier_key {
            for g in 0..self.dev_lru.len() {
                self.dev_lru[g].remove(&(k, eid));
                self.host_lru[g].remove(&(k, eid));
            }
        }
        if self.map.get(&e.hash) == Some(&eid) {
            self.map.remove(&e.hash);
        }
        for (g, &b) in e.blocks.iter().enumerate() {
            if b != NULL_BLOCK {
                self.pools[g].holder[b as usize] = NO_ENTRY;
                self.decref(g, b);
                self.settle(g, b, None);
            }
        }
        for (g, &h) in e.host.iter().enumerate() {
            if h != NULL_BLOCK {
                self.free_host(g, h, false);
            }
        }
        if let Some(p) = e.parent
            && let Some(parent) = self.entries.get_mut(&p)
        {
            parent.children.retain(|c| *c != eid);
            self.lru_refresh(p);
        }
    }

    /// Seals the sequence's next full block: reuses an existing entry for its hash or
    /// creates one holding the sequence's full-attention blocks. Returns false if the
    /// sequence's chain is no longer anchored in the cache (an ancestor was detached).
    fn seal_next(&mut self, seq: &mut SeqKv, id: SeqId, now: Millis) -> bool {
        let idx = seq.entries.len();
        let parent = if idx == 0 {
            None
        } else {
            let p = seq.entries[idx - 1];
            if !self.entries.contains_key(&p) {
                return false;
            }
            Some(p)
        };
        let h = seq.hashes[idx];
        if let Some(&eid) = self.map.get(&h) {
            if self.expired(eid, now) {
                self.remove_subtree(eid);
            } else if self.entries[&eid].parent == parent {
                let e = self.entries.get_mut(&eid).expect("entry");
                e.users += 1;
                e.last_used = now;
                self.lru_refresh(eid);
                seq.entries.push(eid);
                return true;
            } else {
                return false;
            }
        }
        let groups = self.pools.len();
        let eid = self.next_entry;
        self.next_entry += 1;
        let mut blocks = vec![NULL_BLOCK; groups];
        for (g, slot) in blocks.iter_mut().enumerate() {
            if self.attention[g] == AttentionKind::Full {
                let b = seq.blocks[g][idx];
                debug_assert_ne!(b, NULL_BLOCK);
                let pool = &mut self.pools[g];
                debug_assert_eq!(pool.holder[b as usize], NO_ENTRY);
                pool.holder[b as usize] = eid;
                pool.refs[b as usize] += 1;
                *slot = b;
            }
        }
        self.entries.insert(
            eid,
            Entry {
                hash: h,
                parent,
                depth: idx as u32,
                children: Vec::new(),
                users: 1,
                blocks,
                host: vec![NULL_BLOCK; groups],
                created: now,
                last_used: now,
                lru_key: None,
                tier_key: None,
            },
        );
        self.map.insert(h, eid);
        if let Some(p) = parent {
            self.entries.get_mut(&p).expect("parent").children.push(eid);
            self.lru_refresh(p);
        }
        seq.entries.push(eid);
        if !seq.retain {
            self.private_entries.entry(id).or_default().push(eid);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::mimo_like_spec;

    fn salt() -> Lineage {
        lineage(9)
    }

    fn lineage(n: u8) -> Lineage {
        use crate::secret::{BootKey, CacheKey, SaltDeriver};
        SaltDeriver::with_boot_key(BootKey::from_bytes([0; 32]))
            .lineage(&CacheKey::from_bytes([n; 32]))
    }

    #[test]
    fn sliding_blocks_leave_the_window_and_are_zeroed() {
        // Block 4, window 6: at a sealed boundary of 16 the window starts at position 11,
        // so blocks 0 and 1 (positions 0..8) are released.
        let spec = mimo_like_spec(32, 4, 6, 6, 64, 4, 0);
        let mut kv = KvManager::new(&spec, CachePolicy::default());
        let tokens: Vec<u32> = (0..40).collect();
        kv.admit(1, salt(), true, &tokens, 40, 0);
        assert!(kv.allocate(1, 17));
        kv.take_pending();
        kv.commit(1, 17, &tokens, 0);
        let PendingWork {
            maintenance: maint,
            table_updates: tables,
            ..
        } = kv.take_pending();
        let zeroed: Vec<(u32, u32)> = maint
            .iter()
            .filter_map(|m| match *m {
                Maintenance::Zero { group, block } => Some((group, block)),
                _ => None,
            })
            .collect();
        // Groups 1 and 2 are sliding: two blocks each; group 0 keeps everything.
        assert_eq!(zeroed.iter().filter(|z| z.0 == 0).count(), 0);
        assert_eq!(zeroed.iter().filter(|z| z.0 == 1).count(), 2);
        assert_eq!(zeroed.iter().filter(|z| z.0 == 2).count(), 2);
        assert!(tables.iter().all(|t| t.block == NULL_BLOCK && t.index < 2));
        kv.check_invariants();
    }

    #[test]
    fn allocation_failure_leaves_no_trace() {
        let spec = mimo_like_spec(32, 4, 6, 6, 5, 4, 0);
        let mut kv = KvManager::new(&spec, CachePolicy::default());
        let tokens: Vec<u32> = (0..40).collect();
        kv.admit(1, salt(), true, &tokens, 40, 0);
        assert!(kv.allocate(1, 16));
        assert!(!kv.allocate(1, 20));
        kv.check_invariants();
        assert_eq!(kv.free_blocks(0), 0);
        kv.release(1, Release::Preempt, 0);
        kv.check_invariants();
        assert_eq!(kv.free_blocks(0), 4);
    }
}
