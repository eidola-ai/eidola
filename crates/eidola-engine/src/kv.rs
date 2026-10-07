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
//! blocks are attached to entries only where they complete the window of a *hit point*:
//! when a sequence is released, the windows ending at (a) its last sealed boundary (the
//! point a continuation of the conversation resumes from) and (b) the last full block of
//! its prompt before the final token (the point a regeneration of the same prompt resumes
//! from) are retained. A lookup accepts a hit of `m` blocks only if every full-attention
//! block `0..m` is cached and every sliding-window group has its blocks covering
//! `[first_visible(m * block_size), m * block_size)` attached. Hits are therefore always
//! correct, and windows are retained only where they can be used.
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

use std::collections::{BTreeSet, HashMap};

use crate::executor::{Maintenance, Slot, TableUpdate};
use crate::hash::{BlockHash, extend_chain};
use crate::secret::EngineSalt;
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
}

impl Pool {
    fn new(num_blocks: u32) -> Self {
        // Pop from the end: hand out low ids first (purely cosmetic).
        let free = (1..num_blocks).rev().collect();
        Self {
            free,
            refs: vec![0; num_blocks as usize],
        }
    }
}

type EntryId = u64;

#[derive(Debug)]
struct Entry {
    hash: BlockHash,
    parent: Option<EntryId>,
    depth: u32,
    children: Vec<EntryId>,
    users: u32,
    blocks: Vec<u32>,
    created: Millis,
    last_used: Millis,
    lru_key: Option<(Millis, EntryId)>,
}

#[derive(Debug)]
struct SeqKv {
    slot: Slot,
    salt: EngineSalt,
    retain: bool,
    hashes: Vec<BlockHash>,
    entries: Vec<EntryId>,
    chain_broken: bool,
    blocks: Vec<Vec<u32>>,
    computed: u32,
    pins: Vec<(u32, u32)>,
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
        }
    }

    /// The cache policy in force.
    pub fn policy(&self) -> CachePolicy {
        self.policy
    }

    /// Takes the maintenance and table updates accumulated since the last call; they must
    /// be handed to the executor, in order, before or with the next step.
    pub fn take_pending(&mut self) -> (Vec<Maintenance>, Vec<TableUpdate>) {
        (
            std::mem::take(&mut self.pending_maint),
            std::mem::take(&mut self.pending_tables),
        )
    }

    /// Whether any maintenance is waiting for the executor.
    pub fn has_pending_maintenance(&self) -> bool {
        !self.pending_maint.is_empty()
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
    /// point whose window is retained).
    pub fn admit(
        &mut self,
        id: SeqId,
        salt: EngineSalt,
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
            salt,
            retain,
            hashes: Vec::new(),
            entries: Vec::new(),
            chain_broken: false,
            blocks: vec![Vec::new(); groups],
            computed: 0,
            pins: vec![(0, 0); groups],
            slide_cursor: vec![0; groups],
        };
        let regen_point = (prompt_len.max(1) - 1) / bs;
        for g in 0..groups {
            if let AttentionKind::Sliding { .. } = self.attention[g] {
                let lo = self.attention[g].first_visible(regen_point * bs) / bs;
                seq.pins[g] = (lo, regen_point);
            }
        }

        if self.policy.enabled {
            let cap_blocks = (tokens.len() as u32 - 1) / bs;
            let mut hashes = Vec::new();
            extend_chain(
                &seq.salt,
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
            let mut m = chain.len() as u32;
            while m > 0 && !self.window_covered(&chain, m) {
                m -= 1;
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
                        cached
                    } else {
                        NULL_BLOCK
                    };
                    if b != NULL_BLOCK {
                        self.pools[g].refs[b as usize] += 1;
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
                &seq.salt,
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

        // Slide windows behind the newest sealed boundary.
        for g in 0..groups {
            if let AttentionKind::Sliding { .. } = self.attention[g] {
                let lo_keep = self.attention[g].first_visible(sealable * bs) / bs;
                let (pin_lo, pin_hi) = seq.pins[g];
                while seq.slide_cursor[g] < lo_keep {
                    let idx = seq.slide_cursor[g];
                    seq.slide_cursor[g] += 1;
                    if idx >= pin_lo && idx < pin_hi {
                        continue;
                    }
                    if (idx as usize) < seq.blocks[g].len() {
                        self.unmap(&mut seq, g, idx as usize);
                    }
                }
            }
        }
        self.seqs.insert(id, seq);
    }

    /// Releases a sequence's slot and blocks. See [`Release`].
    pub fn release(&mut self, id: SeqId, kind: Release, prompt_len: u32, now: Millis) {
        let bs = self.block_size;
        let groups = self.pools.len();
        let seq = self.seqs.remove(&id).expect("live sequence");
        let retain = kind == Release::Preempt || seq.retain;

        if self.policy.enabled && retain {
            let sealed = seq.entries.len() as u32;
            let regen = (prompt_len.max(1) - 1) / bs;
            for point in [regen, sealed] {
                if point == 0 || point > sealed {
                    continue;
                }
                for g in 0..groups {
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
                            && entry.blocks[g] == NULL_BLOCK
                        {
                            entry.blocks[g] = b;
                            self.pools[g].refs[b as usize] += 1;
                        }
                    }
                }
            }
        }

        for g in 0..groups {
            for idx in 0..seq.blocks[g].len() {
                let b = seq.blocks[g][idx];
                if b != NULL_BLOCK {
                    self.decref(g, b);
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
        }
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
            for (g, &b) in e.blocks.iter().enumerate() {
                if self.attention[g] == AttentionKind::Full {
                    assert_ne!(b, NULL_BLOCK, "entry lacks a full-attention block");
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
    }

    fn decref(&mut self, g: usize, b: u32) {
        let r = &mut self.pools[g].refs[b as usize];
        assert!(*r > 0, "group {g}: decref of free block {b}");
        *r -= 1;
        if *r == 0 {
            self.pools[g].free.push(b);
            self.pending_maint.push(Maintenance::Zero {
                group: g as u32,
                block: b,
            });
        }
    }

    fn alloc_block(&mut self, g: usize) -> Option<u32> {
        loop {
            if let Some(b) = self.pools[g].free.pop() {
                self.pools[g].refs[b as usize] = 1;
                return Some(b);
            }
            if !self.evict_one() {
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
                if self.entries[&eid].blocks[g] == NULL_BLOCK {
                    return false;
                }
            }
        }
        true
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
        if self.map.get(&e.hash) == Some(&eid) {
            self.map.remove(&e.hash);
        }
        for (g, &b) in e.blocks.iter().enumerate() {
            if b != NULL_BLOCK {
                self.decref(g, b);
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
        let mut blocks = vec![NULL_BLOCK; groups];
        for (g, slot) in blocks.iter_mut().enumerate() {
            if self.attention[g] == AttentionKind::Full {
                let b = seq.blocks[g][idx];
                debug_assert_ne!(b, NULL_BLOCK);
                self.pools[g].refs[b as usize] += 1;
                *slot = b;
            }
        }
        let eid = self.next_entry;
        self.next_entry += 1;
        self.entries.insert(
            eid,
            Entry {
                hash: h,
                parent,
                depth: idx as u32,
                children: Vec::new(),
                users: 1,
                blocks,
                created: now,
                last_used: now,
                lru_key: None,
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

    fn salt() -> EngineSalt {
        EngineSalt::from_bytes([9; 32])
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
        let (maint, tables) = kv.take_pending();
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
        kv.release(1, Release::Preempt, 40, 0);
        kv.check_invariants();
        assert_eq!(kv.free_blocks(0), 4);
    }
}
