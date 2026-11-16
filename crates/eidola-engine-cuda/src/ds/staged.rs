//! Dev-only: the whole model on one GPU, its backbone's routed experts cached on the
//! device from host memory.
//!
//! The released model is about 510 GB; one B300 holds about 288. Without the backbone's
//! routed experts (289 GB with their scales) everything else, both Engram tables
//! included, is about 222 GB, which leaves room on one device for a cache of a few
//! thousand of the 15,360 experts at 18.8 MB each. Each half keeps its own cache
//! ([`ExpertCache`]): a device array per expert-major weight array, `capacity` experts
//! long, with a host index ([`ExpertIndex`]) of which (layer, expert) each slot holds.
//!
//! A staged feed-forward runs the router (the production launch), reads the selected
//! ids back, and splits the rows into rounds whose experts fit the cache
//! ([`plan_rounds`]). Per round the missing experts are copied from host memory into
//! slots (the least recently used evicted, never one the round needs), the round's ids
//! are rewritten to slot numbers, and the placement and both grouped products run over
//! the cache with `capacity` groups; the round is then combined with the shared expert
//! before the next round reuses the layout. Every kernel is the production one and
//! computes each pair from its own row and its expert's weights alone, so the outputs
//! are the unstaged executor's bit for bit; only weight addresses, the layout's row
//! order and the launches' row counts differ. The cost is a read-back per layer and the
//! copies.
//!
//! **Never part of a pinned deployment.** This module exists only with the crate's
//! `dev-staged-experts` feature, which the node's image build does not enable; every
//! constructor takes a [`DevOnly`] token, built only by tests and development examples;
//! and nothing in the node's configuration can name it. `tests/ds_staged_release.rs`
//! holds all three.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::DsConfig;
use eidola_engine_model::safetensors::Dtype;

use super::executor::{DsCudaExecutor, DsCudaExecutorConfig};
use super::half::{Buffer, CudaDevice, Device, Kind};
use super::model::{HalfNet, Routing};
use super::support::{DsShape, check_supported};
use super::weights::{Experts, Half, HalfWeights};
use crate::module::KernelDir;
use crate::{CudaError, Result, narrow};

/// Proof that the caller is a test or a development tool. Every constructor of this
/// module takes one.
#[derive(Clone, Copy, Debug)]
pub struct DevOnly(());

impl DevOnly {
    /// For tests and development examples only: no source a node runs may call it
    /// (`tests/ds_staged_release.rs` holds every other source of the workspace to that).
    pub fn for_tests_and_dev_tools() -> DevOnly {
        DevOnly(())
    }
}

/// An absent entry of the index.
const NONE: u32 = u32::MAX;

/// What one admission did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Admission {
    /// Per id admitted, the slot holding its expert.
    pub slots: Vec<u32>,
    /// `(slot, expert)` to copy in, each slot once.
    pub loads: Vec<(u32, u32)>,
    /// `(layer, expert)` no longer held.
    pub evicted: Vec<(u32, u32)>,
    /// Distinct experts already held.
    pub hits: u32,
}

/// Which (layer, expert) each of `capacity` slots holds, least recently admitted first
/// out. Pure host state.
#[derive(Clone, Debug)]
pub struct ExpertIndex {
    layers: u32,
    experts: u32,
    capacity: u32,
    /// `[layer · experts + expert]`: its slot, or [`NONE`].
    slot_of: Vec<u32>,
    /// `[slot]`: the key held, or [`NONE`].
    owner: Vec<u32>,
    /// `[slot]`: the admission that last used it (0: never).
    used: Vec<u64>,
    /// `(used, slot)` of every slot: the eviction order.
    order: BTreeSet<(u64, u32)>,
    tick: u64,
}

impl ExpertIndex {
    pub fn new(layers: u32, experts: u32, capacity: u32) -> Result<ExpertIndex> {
        let keys = layers
            .checked_mul(experts)
            .filter(|k| *k < NONE)
            .ok_or_else(|| CudaError::new("staged experts: too many layers and experts"))?;
        if capacity == 0 || capacity == NONE {
            return Err(CudaError::new("staged experts: a cache of no slot"));
        }
        Ok(ExpertIndex {
            layers,
            experts,
            capacity,
            slot_of: vec![NONE; keys as usize],
            owner: vec![NONE; capacity as usize],
            used: vec![0; capacity as usize],
            order: (0..capacity).map(|s| (0, s)).collect(),
            tick: 0,
        })
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The slot holding `layer`'s `expert`, if any.
    pub fn slot(&self, layer: u32, expert: u32) -> Option<u32> {
        let s = self.slot_of[self.key(layer, expert) as usize];
        (s != NONE).then_some(s)
    }

    /// Experts held.
    pub fn held(&self) -> usize {
        self.owner.iter().filter(|o| **o != NONE).count()
    }

    fn key(&self, layer: u32, expert: u32) -> u32 {
        layer * self.experts + expert
    }

    fn touch(&mut self, slot: u32) {
        let s = slot as usize;
        self.order.remove(&(self.used[s], slot));
        self.used[s] = self.tick;
        self.order.insert((self.tick, slot));
    }

    /// Brings every expert `ids` names (of `layer`) into a slot and maps each id to it.
    /// A missing expert takes the least recently used slot not used by this admission
    /// (a never-used slot first, then the lowest); refused when the distinct experts
    /// outnumber the slots, or an id or the layer is out of range, before anything
    /// changes.
    pub fn admit(&mut self, layer: u32, ids: &[u32]) -> Result<Admission> {
        if layer >= self.layers {
            return Err(CudaError::new(format!(
                "staged experts: layer {layer} of {}",
                self.layers
            )));
        }
        if let Some(id) = ids.iter().find(|id| **id >= self.experts) {
            return Err(CudaError::new(format!(
                "staged experts: expert {id} of {}",
                self.experts
            )));
        }
        let mut distinct = ids.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() > self.capacity as usize {
            return Err(CudaError::new(format!(
                "staged experts: {} distinct experts for a cache of {}",
                distinct.len(),
                self.capacity
            )));
        }
        self.tick += 1;
        let mut out = Admission::default();
        let mut missing = Vec::new();
        for &e in &distinct {
            match self.slot(layer, e) {
                Some(slot) => {
                    self.touch(slot);
                    out.hits += 1;
                }
                None => missing.push(e),
            }
        }
        for e in missing {
            let &(used, slot) = self.order.first().expect("a slot");
            assert!(
                used < self.tick,
                "every slot is used by this admission, which holds no more than the cache"
            );
            let old = self.owner[slot as usize];
            if old != NONE {
                self.slot_of[old as usize] = NONE;
                out.evicted.push((old / self.experts, old % self.experts));
            }
            let key = self.key(layer, e);
            self.owner[slot as usize] = key;
            self.slot_of[key as usize] = slot;
            self.touch(slot);
            out.loads.push((slot, e));
        }
        out.slots = ids
            .iter()
            .map(|&e| self.slot(layer, e).expect("admitted"))
            .collect();
        Ok(out)
    }
}

/// Rounds of consecutive rows (`(first, count)`) covering `ids` (`[rows, active]`), each
/// as long as its distinct experts fit `capacity`. Refused when one row's do not.
pub fn plan_rounds(ids: &[u32], active: u32, capacity: u32) -> Result<Vec<(u32, u32)>> {
    let a = active as usize;
    if a == 0 || !ids.len().is_multiple_of(a) {
        return Err(CudaError::new("staged experts: ids are not whole rows"));
    }
    let rows: u32 = narrow(ids.len() / a, "rows")?;
    let mut rounds = Vec::new();
    let (mut first, mut held) = (0u32, HashSet::new());
    for r in 0..rows {
        let row = &ids[r as usize * a..(r as usize + 1) * a];
        let mut with: HashSet<u32> = held.clone();
        with.extend(row.iter().copied());
        if with.len() > capacity as usize {
            let own: HashSet<u32> = row.iter().copied().collect();
            if own.len() > capacity as usize {
                return Err(CudaError::new(format!(
                    "staged experts: a row's {} experts do not fit a cache of {capacity}",
                    own.len()
                )));
            }
            rounds.push((first, r - first));
            first = r;
            with = own;
        }
        held = with;
    }
    if rows > first {
        rounds.push((first, rows - first));
    }
    Ok(rounds)
}

/// How a run reads the experts on a miss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostMode {
    /// Straight from the mapped checkpoint, every time (the page cache keeps what it can).
    Mapped,
    /// Each expert read once from the checkpoint into host memory on `threads` threads,
    /// in the device's expert-major order, and copied from there.
    Read { threads: usize },
}

/// Where the experts' bytes are.
enum Source {
    Mapped(Arc<WeightSet>),
    /// Per backbone layer, `[experts, per expert]` in the device's order.
    Read(Vec<Vec<u8>>),
}

/// The backbone's routed experts in host memory.
pub struct HostExperts {
    source: Source,
    /// Each layer's tensor prefix (`layers.<i>`).
    prefixes: Vec<String>,
    experts: u32,
    inter: usize,
    dim: usize,
    /// Per expert: bytes of the up array (gate rows, then up rows), its scales, the down
    /// array, its scales.
    sizes: [usize; 4],
    /// Seconds and bytes the reading took (0 when mapped).
    pub read_seconds: f64,
    pub read_bytes: u64,
}

/// One part of one expert: `(array, offset in the expert's run, tensor, dtype, shape)`.
type Part = (usize, usize, String, Dtype, [usize; 2]);

impl HostExperts {
    /// The experts of `shape`'s backbone layers in `store`, each tensor's dtype and shape
    /// checked.
    pub fn new(
        store: &Arc<WeightSet>,
        c: &DsConfig,
        shape: &DsShape,
        mode: HostMode,
        _: DevOnly,
    ) -> Result<HostExperts> {
        let (d, inter) = (c.dim, c.moe.inter_dim);
        let prefixes: Vec<String> = (0..shape.layers)
            .map(|l| format!("layers.{}", c.layers[l].source_index))
            .collect();
        let mut host = HostExperts {
            source: Source::Mapped(store.clone()),
            prefixes,
            experts: narrow(c.moe.n_routed, "experts")?,
            inter,
            dim: d,
            sizes: [
                2 * (inter * d / 2),
                2 * (inter * d / 32),
                d * inter / 2,
                d * inter / 32,
            ],
            read_seconds: 0.0,
            read_bytes: 0,
        };
        for l in 0..shape.layers {
            for e in 0..host.experts {
                for (_, _, name, dtype, shape) in host.parts(l, e) {
                    let t = store
                        .get(&name)
                        .map_err(|e| CudaError::new(e.to_string()))?;
                    t.expect(&name, dtype, &shape)
                        .map_err(|e| CudaError::new(e.to_string()))?;
                }
            }
        }
        if let HostMode::Read { threads } = mode {
            host.read(store, threads.max(1))?;
        }
        Ok(host)
    }

    /// Bytes of one expert.
    pub fn expert_bytes(&self) -> usize {
        self.sizes.iter().sum()
    }

    /// Bytes of each of the expert's four device arrays.
    pub fn sizes(&self) -> [usize; 4] {
        self.sizes
    }

    pub fn layers(&self) -> usize {
        self.prefixes.len()
    }

    pub fn experts(&self) -> u32 {
        self.experts
    }

    fn parts(&self, l: usize, e: u32) -> [Part; 6] {
        let (d, inter) = (self.dim, self.inter);
        let p = &self.prefixes[l];
        let name = |w: &str, part: &str| format!("{p}.ffn.experts.{e}.{w}.{part}");
        let (up, ups) = (self.sizes[0] / 2, self.sizes[1] / 2);
        [
            (0, 0, name("w1", "weight"), Dtype::I8, [inter, d / 2]),
            (0, up, name("w3", "weight"), Dtype::I8, [inter, d / 2]),
            (1, 0, name("w1", "scale"), Dtype::F8E8M0, [inter, d / 32]),
            (1, ups, name("w3", "scale"), Dtype::F8E8M0, [inter, d / 32]),
            (2, 0, name("w2", "weight"), Dtype::I8, [d, inter / 2]),
            (3, 0, name("w2", "scale"), Dtype::F8E8M0, [d, inter / 32]),
        ]
    }

    /// Reads every expert once into host memory, on `threads` threads each taking a
    /// contiguous run of experts (so each reads its files in order).
    fn read(&mut self, store: &Arc<WeightSet>, threads: usize) -> Result<()> {
        let began = Instant::now();
        let per = self.expert_bytes();
        let n = self.experts as usize;
        let mut layers: Vec<Vec<u8>> = (0..self.layers()).map(|_| vec![0u8; n * per]).collect();
        let total = (self.layers() * n) as u64;
        let done = AtomicU64::new(0);
        let failed = AtomicUsize::new(0);
        let errors = std::sync::Mutex::new(Vec::new());
        {
            let mut jobs: Vec<(usize, u32, &mut [u8])> = Vec::with_capacity(self.layers() * n);
            for (l, layer) in layers.iter_mut().enumerate() {
                for (e, run) in layer.chunks_exact_mut(per).enumerate() {
                    jobs.push((l, u32::try_from(e).expect("an expert"), run));
                }
            }
            let chunk = jobs.len().div_ceil(threads).max(1);
            let this = &*self;
            std::thread::scope(|scope| {
                for part in jobs.chunks_mut(chunk) {
                    let (done, failed, errors) = (&done, &failed, &errors);
                    scope.spawn(move || {
                        for (l, e, run) in part.iter_mut() {
                            if failed.load(Ordering::Relaxed) > 0 {
                                return;
                            }
                            let mut offsets = [0usize; 4];
                            for i in 1..4 {
                                offsets[i] = offsets[i - 1] + this.sizes[i - 1];
                            }
                            for (array, at, name, _, _) in this.parts(*l, *e) {
                                match store.get(&name) {
                                    Ok(t) => {
                                        let to = offsets[array] + at;
                                        run[to..to + t.data.len()].copy_from_slice(t.data);
                                    }
                                    Err(err) => {
                                        failed.fetch_add(1, Ordering::Relaxed);
                                        errors.lock().expect("lock").push(err.to_string());
                                        return;
                                    }
                                }
                            }
                            let now = done.fetch_add(1, Ordering::Relaxed) + 1;
                            if now * 20 / total != (now - 1) * 20 / total {
                                eprintln!(
                                    "staged experts: {now} of {total} read into host memory \
                                     ({:.0} s)",
                                    began.elapsed().as_secs_f64()
                                );
                            }
                        }
                    });
                }
            });
        }
        if let Some(e) = errors.into_inner().expect("lock").into_iter().next() {
            return Err(CudaError::new(e));
        }
        self.read_seconds = began.elapsed().as_secs_f64();
        self.read_bytes = total * per as u64;
        self.source = Source::Read(layers);
        Ok(())
    }

    /// The copies that bring `l`'s expert `e` into `slot` of `arrays`.
    fn copy_in(
        &self,
        dev: &dyn Device,
        arrays: &[Buffer; 4],
        l: usize,
        e: u32,
        slot: u32,
    ) -> Result<u64> {
        let slot = slot as usize;
        match &self.source {
            Source::Read(layers) => {
                let per = self.expert_bytes();
                let run = &layers[l][e as usize * per..(e as usize + 1) * per];
                let mut at = 0;
                for (array, size) in arrays.iter().zip(self.sizes) {
                    array.write(dev, slot * size, &run[at..at + size])?;
                    at += size;
                }
            }
            Source::Mapped(store) => {
                for (array, offset, name, _, _) in self.parts(l, e) {
                    let t = store
                        .get(&name)
                        .map_err(|e| CudaError::new(e.to_string()))?;
                    arrays[array].write(dev, slot * self.sizes[array] + offset, t.data)?;
                }
            }
        }
        Ok(self.expert_bytes() as u64)
    }
}

/// What a half's cache has done.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StagedStats {
    /// Staged feed-forwards run, and the rounds they took.
    pub layers: u64,
    pub rounds: u64,
    /// Distinct experts a round needed, those already held, and those copied in.
    pub needed: u64,
    pub hits: u64,
    pub loads: u64,
    pub evictions: u64,
    pub bytes_loaded: u64,
    /// Host seconds spent reading ids back (which waits for the device) and queuing
    /// the copies.
    pub readback_seconds: f64,
    pub load_seconds: f64,
}

struct CacheState {
    index: ExpertIndex,
    stats: StagedStats,
}

/// One half's expert cache (module documentation).
pub struct ExpertCache {
    pub capacity: u32,
    /// The model layers this half runs.
    layers: Range<usize>,
    /// `[capacity, ..]` each: the expert-major arrays of `Experts`.
    arrays: [Buffer; 4],
    /// `[capacity]` words each: the group table and the placement's scratch.
    group_end: Buffer,
    place_scratch: Buffer,
    host: Arc<HostExperts>,
    state: RefCell<CacheState>,
}

impl ExpertCache {
    /// A cache of `capacity` experts on `dev` for the model layers `layers`.
    pub fn new(
        dev: &dyn Device,
        host: Arc<HostExperts>,
        layers: Range<usize>,
        capacity: u32,
        active: u32,
        _: DevOnly,
    ) -> Result<ExpertCache> {
        if capacity < active {
            return Err(CudaError::new(format!(
                "staged experts: a cache of {capacity} cannot hold one row's {active} experts"
            )));
        }
        if layers.end > host.layers() {
            return Err(CudaError::new(
                "staged experts: a layer the host does not hold",
            ));
        }
        let c = capacity as usize;
        let s = host.sizes();
        let a = |name, bytes: usize| Buffer::new(dev, name, Kind::Weights, c * bytes);
        let arrays = [
            a("staged up", s[0])?,
            a("staged up scales", s[1])?,
            a("staged down", s[2])?,
            a("staged down scales", s[3])?,
        ];
        let index = ExpertIndex::new(narrow(layers.len(), "layers")?, host.experts(), capacity)?;
        Ok(ExpertCache {
            capacity,
            arrays,
            group_end: Buffer::new(dev, "staged group_end", Kind::Scratch, c * 4)?,
            place_scratch: Buffer::new(dev, "staged place_scratch", Kind::Scratch, c * 4)?,
            layers,
            host,
            state: RefCell::new(CacheState {
                index,
                stats: StagedStats::default(),
            }),
        })
    }

    /// The cache's scratch: the group table and the placement's scratch.
    pub fn scratch(&self) -> [&Buffer; 2] {
        [&self.group_end, &self.place_scratch]
    }

    /// Bytes of the cached weights.
    pub fn weight_bytes(&self) -> u64 {
        self.arrays.iter().map(|b| b.bytes as u64).sum()
    }

    pub fn stats(&self) -> StagedStats {
        self.state.borrow().stats
    }

    /// The rounds of a feed-forward whose routed ids (`[rows, active]`) were read back,
    /// `seconds` after the read-back was asked for.
    pub(crate) fn rounds(&self, ids: &[u32], active: u32, seconds: f64) -> Result<Vec<(u32, u32)>> {
        let rounds = plan_rounds(ids, active, self.capacity)?;
        let mut st = self.state.borrow_mut();
        st.stats.layers += 1;
        st.stats.rounds += rounds.len() as u64;
        st.stats.readback_seconds += seconds;
        Ok(rounds)
    }

    /// Brings the experts `ids` name (of model layer `layer`) into the cache: returns
    /// each id's slot and where the round's pairs find their experts.
    pub(crate) fn admit(
        &self,
        dev: &dyn Device,
        layer: usize,
        ids: &[u32],
    ) -> Result<(Vec<u32>, Routing)> {
        let began = Instant::now();
        let local = layer
            .checked_sub(self.layers.start)
            .filter(|l| *l < self.layers.len())
            .ok_or_else(|| {
                CudaError::new(format!("staged experts: layer {layer} of another half"))
            })?;
        let mut st = self.state.borrow_mut();
        let a = st.index.admit(narrow(local, "layer")?, ids)?;
        for &(slot, e) in &a.loads {
            st.stats.bytes_loaded += self.host.copy_in(dev, &self.arrays, layer, e, slot)?;
        }
        let s = &mut st.stats;
        s.needed += u64::from(a.hits) + a.loads.len() as u64;
        s.hits += u64::from(a.hits);
        s.loads += a.loads.len() as u64;
        s.evictions += a.evicted.len() as u64;
        s.load_seconds += began.elapsed().as_secs_f64();
        let routing = Routing {
            groups: self.capacity,
            group_end: self.group_end.addr,
            scratch: self.place_scratch.addr,
            experts: Experts {
                up_packed: self.arrays[0].addr,
                up_scales: self.arrays[1].addr,
                down_packed: self.arrays[2].addr,
                down_scales: self.arrays[3].addr,
            },
        };
        Ok((a.slots, routing))
    }
}

/// How large each half's cache is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capacity {
    /// Experts per half: the encoder's, the decoder's.
    Experts([u32; 2]),
    /// Every byte the device has free once the halves are loaded, but `reserve`, split
    /// evenly when both halves share a device.
    Fill { reserve: usize },
}

/// The model layers `half` runs.
pub fn layers_of(shape: &DsShape, half: Half) -> Range<usize> {
    match half {
        Half::Encoder => 0..shape.split,
        Half::Decoder => shape.split..shape.layers,
    }
}

/// A half's weights without the backbone's routed experts (`HalfWeights::load` otherwise,
/// the Engram tables of its device included).
pub fn half_weights(
    dev: &dyn Device,
    store: Option<&WeightSet>,
    c: &DsConfig,
    shape: &DsShape,
    half: Half,
    max_model_len: u32,
    _: DevOnly,
) -> Result<HalfWeights> {
    HalfWeights::load_with(dev, store, c, shape, half, max_model_len, true, false)
}

/// Gives `n` (laid out with [`half_weights`]) a cache of `capacity` experts.
pub fn attach(
    n: &mut HalfNet,
    shape: &DsShape,
    host: &Arc<HostExperts>,
    capacity: u32,
    token: DevOnly,
) -> Result<()> {
    let cache = ExpertCache::new(
        n.dev.as_ref(),
        host.clone(),
        layers_of(shape, n.half),
        capacity,
        n.d.moe.active,
        token,
    )?;
    n.staged = Some(cache);
    Ok(())
}

/// Bytes free on device `ordinal` now.
pub fn free_device_bytes(ordinal: usize) -> Result<usize> {
    let ctx = cudarc::driver::CudaContext::new(ordinal)?;
    Ok(ctx.mem_get_info()?.0)
}

/// Experts per half a [`Capacity`] comes to, given each half's free bytes (`shared`:
/// both halves on one device, whose free bytes are the first).
pub fn capacities(
    capacity: Capacity,
    free: [usize; 2],
    shared: bool,
    expert_bytes: usize,
) -> Result<[u32; 2]> {
    match capacity {
        Capacity::Experts(n) => Ok(n),
        Capacity::Fill { reserve } => {
            let per = |free: usize, halves: usize| -> Result<u32> {
                let room = free.checked_sub(reserve).ok_or_else(|| {
                    CudaError::new(format!(
                        "staged experts: {free} bytes free, under the reserve of {reserve}"
                    ))
                })?;
                // Each expert also takes two words of the group table and scratch.
                narrow(room / halves / (expert_bytes + 8), "cache slots")
            };
            if shared {
                let n = per(free[0], 2)?;
                Ok([n, n])
            } else {
                Ok([per(free[0], 1)?, per(free[1], 1)?])
            }
        }
    }
}

impl DsCudaExecutor {
    /// The executor on CUDA devices with the backbone's routed experts staged from
    /// `host` (module documentation).
    #[allow(clippy::too_many_arguments)]
    pub fn staged(
        dir: &KernelDir,
        store: &Arc<WeightSet>,
        c: &DsConfig,
        token_map: Option<Vec<u32>>,
        cfg: &DsCudaExecutorConfig,
        host: &Arc<HostExperts>,
        capacity: Capacity,
        token: DevOnly,
    ) -> Result<DsCudaExecutor> {
        Self::preflight(c, cfg)?;
        let encoder = Box::new(CudaDevice::open(cfg.devices[0], dir)?);
        let decoder = Box::new(CudaDevice::open(cfg.devices[1], dir)?);
        let mut x = Self::build_with(encoder, decoder, Some(store), c, token_map, cfg, false)?;
        let shared = cfg.devices[0] == cfg.devices[1];
        let free = [
            free_device_bytes(cfg.devices[0])?,
            free_device_bytes(cfg.devices[1])?,
        ];
        let n = capacities(capacity, free, shared, host.expert_bytes())?;
        x.attach_caches(&check_supported(c, cfg.draft_width)?, host, n, token)?;
        Ok(x)
    }

    /// As [`DsCudaExecutor::staged`], over two given devices with caches of `capacity`
    /// experts (the encoder's, the decoder's).
    #[allow(clippy::too_many_arguments)]
    pub fn staged_with_devices(
        encoder: Box<dyn Device>,
        decoder: Box<dyn Device>,
        store: &Arc<WeightSet>,
        c: &DsConfig,
        token_map: Option<Vec<u32>>,
        cfg: &DsCudaExecutorConfig,
        host: &Arc<HostExperts>,
        capacity: [u32; 2],
        token: DevOnly,
    ) -> Result<DsCudaExecutor> {
        let mut x = Self::build_with(encoder, decoder, Some(store), c, token_map, cfg, false)?;
        x.attach_caches(&check_supported(c, cfg.draft_width)?, host, capacity, token)?;
        Ok(x)
    }

    fn attach_caches(
        &mut self,
        shape: &DsShape,
        host: &Arc<HostExperts>,
        capacity: [u32; 2],
        token: DevOnly,
    ) -> Result<()> {
        for (n, cap) in self.halves_mut().into_iter().zip(capacity) {
            attach(n, shape, host, cap, token)?;
        }
        Ok(())
    }

    /// Each half's cache statistics: the encoder's, the decoder's.
    pub fn staged_stats(&self) -> [StagedStats; 2] {
        [Half::Encoder, Half::Decoder].map(|h| {
            self.half(h)
                .staged
                .as_ref()
                .map(ExpertCache::stats)
                .unwrap_or_default()
        })
    }

    /// Each half's cache size in experts (0 without one).
    pub fn staged_capacity(&self) -> [u32; 2] {
        [Half::Encoder, Half::Decoder]
            .map(|h| self.half(h).staged.as_ref().map_or(0, |c| c.capacity))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(layers: u32, experts: u32, capacity: u32) -> ExpertIndex {
        ExpertIndex::new(layers, experts, capacity).unwrap()
    }

    #[test]
    fn misses_fill_free_slots_then_evict_the_least_recently_used() {
        let mut x = index(2, 8, 3);
        let a = x.admit(0, &[4, 1]).unwrap();
        // Ascending expert order takes the lowest free slots.
        assert_eq!(a.loads, vec![(0, 1), (1, 4)]);
        assert_eq!(a.slots, vec![1, 0]);
        assert_eq!((a.hits, a.evicted.len()), (0, 0));
        let a = x.admit(1, &[1]).unwrap();
        assert_eq!(
            a.loads,
            vec![(2, 1)],
            "another layer's expert 1 is another key"
        );
        // Full: layer 0's experts 1 and 4 (tick 1) are older than layer 1's 1 (tick 2);
        // of the two, slot 0 first.
        let a = x.admit(0, &[7]).unwrap();
        assert_eq!(a.loads, vec![(0, 7)]);
        assert_eq!(a.evicted, vec![(0, 1)]);
        assert_eq!(x.slot(0, 1), None);
        assert_eq!(x.slot(0, 4), Some(1));
        let a = x.admit(0, &[4, 4, 7]).unwrap();
        assert_eq!((a.hits, a.loads.len()), (2, 0));
        assert_eq!(a.slots, vec![1, 1, 0]);
        // Now layer 1's expert 1 is the oldest.
        let a = x.admit(1, &[3]).unwrap();
        assert_eq!(a.evicted, vec![(1, 1)]);
        assert_eq!(a.loads, vec![(2, 3)]);
    }

    #[test]
    fn an_admission_never_evicts_an_expert_it_needs() {
        let mut x = index(1, 16, 2);
        x.admit(0, &[0, 1]).unwrap();
        x.admit(0, &[0]).unwrap();
        // 1 is the least recently used, but 1 is needed: 0 goes.
        let a = x.admit(0, &[2, 1]).unwrap();
        assert_eq!(a.evicted, vec![(0, 0)]);
        assert_eq!(a.hits, 1);
        assert_eq!(x.slot(0, 1), Some(1));
        assert_eq!(x.slot(0, 2), Some(0));
        // Exactly the capacity, every slot replaced.
        let a = x.admit(0, &[9, 8, 9]).unwrap();
        assert_eq!(a.loads.len(), 2);
        assert_eq!(a.evicted.len(), 2);
        assert_eq!(a.slots[0], a.slots[2]);
        assert_ne!(a.slots[0], a.slots[1]);
    }

    #[test]
    fn what_does_not_fit_is_refused_before_anything_changes() {
        let mut x = index(2, 8, 2);
        x.admit(0, &[3]).unwrap();
        let before = (x.slot_of.clone(), x.owner.clone(), x.used.clone(), x.tick);
        assert!(x.admit(0, &[0, 1, 2]).is_err());
        assert!(x.admit(0, &[8]).is_err());
        assert!(x.admit(2, &[0]).is_err());
        assert_eq!(
            before,
            (x.slot_of.clone(), x.owner.clone(), x.used.clone(), x.tick)
        );
        assert!(ExpertIndex::new(1, 8, 0).is_err());
        // One slot: every admission of a new expert replaces it.
        let mut one = index(1, 4, 1);
        for e in [0, 1, 1, 2, 0] {
            let a = one.admit(0, &[e, e]).unwrap();
            assert_eq!(a.slots, vec![0, 0]);
        }
        assert_eq!(one.held(), 1);
        // Nothing admitted: nothing changes but the clock.
        assert_eq!(one.admit(0, &[]).unwrap(), Admission::default());
    }

    /// Over many seeded admissions, against each held expert's last admission: a miss
    /// evicts only when the cache is full, never an expert the admission needs, and only
    /// experts used no later than every one it keeps; every id maps to a slot holding
    /// it, and no slot is held twice.
    #[test]
    fn the_index_is_an_lru_cache() {
        use std::collections::HashMap;
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        for (layers, experts, capacity) in [(3u32, 16u32, 5u32), (2, 6, 6), (4, 32, 20), (1, 9, 2)]
        {
            let mut x = index(layers, experts, capacity);
            // Held (layer, expert) -> the admission that last used it.
            let mut last: HashMap<(u32, u32), u64> = HashMap::new();
            let mut evictions = 0;
            for tick in 1..=3000u64 {
                let layer = u32::try_from(next(u64::from(layers))).unwrap();
                let n = 1 + next(u64::from(capacity) + 2);
                let ids: Vec<u32> = (0..n)
                    .map(|_| u32::try_from(next(u64::from(experts))).unwrap())
                    .collect();
                let mut distinct = ids.clone();
                distinct.sort_unstable();
                distinct.dedup();
                let Ok(a) = x.admit(layer, &ids) else {
                    assert!(distinct.len() > capacity as usize);
                    continue;
                };
                assert!(distinct.len() <= capacity as usize);
                let missing: Vec<u32> = distinct
                    .iter()
                    .copied()
                    .filter(|e| !last.contains_key(&(layer, *e)))
                    .collect();
                let mut loaded: Vec<u32> = a.loads.iter().map(|l| l.1).collect();
                loaded.sort_unstable();
                assert_eq!(loaded, missing, "exactly the missing experts are loaded");
                let over = (last.len() + missing.len()).saturating_sub(capacity as usize);
                assert_eq!(a.evicted.len(), over, "evictions only when full");
                let kept_oldest = last
                    .iter()
                    .filter(|(k, _)| k.0 != layer || !distinct.contains(&k.1))
                    .filter(|(k, _)| !a.evicted.contains(k))
                    .map(|(_, t)| *t)
                    .min()
                    .unwrap_or(u64::MAX);
                for k in &a.evicted {
                    assert!(
                        k.0 != layer || !distinct.contains(&k.1),
                        "a needed expert evicted"
                    );
                    assert!(
                        last[k] <= kept_oldest,
                        "{k:?} evicted before an older expert"
                    );
                    last.remove(k);
                }
                evictions += a.evicted.len();
                for &e in &distinct {
                    last.insert((layer, e), tick);
                }
                assert_eq!(x.held(), last.len());
                for k in last.keys() {
                    assert!(x.slot(k.0, k.1).is_some(), "{k:?} held");
                }
                for (id, slot) in ids.iter().zip(&a.slots) {
                    assert_eq!(x.owner[*slot as usize], x.key(layer, *id));
                }
                let held: HashSet<u32> = x.owner.iter().copied().filter(|o| *o != NONE).collect();
                assert_eq!(held.len(), x.held());
            }
            assert!(evictions > 100, "the workload evicts");
        }
    }

    #[test]
    fn rounds_are_the_longest_runs_of_rows_whose_experts_fit() {
        // Rows of two experts.
        let ids = [0, 1, 1, 2, 3, 4, 0, 1, 5, 6];
        assert_eq!(plan_rounds(&ids, 2, 8).unwrap(), vec![(0, 5)]);
        assert_eq!(
            plan_rounds(&ids, 2, 3).unwrap(),
            vec![(0, 2), (2, 1), (3, 1), (4, 1)]
        );
        assert_eq!(
            plan_rounds(&ids, 2, 4).unwrap(),
            vec![(0, 2), (2, 2), (4, 1)]
        );
        assert_eq!(
            plan_rounds(&ids, 2, 2).unwrap(),
            vec![(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
        );
        // A row's own experts must fit; a row repeating an expert counts it once.
        assert!(plan_rounds(&ids, 2, 1).is_err());
        assert_eq!(plan_rounds(&[3, 3, 3, 3], 2, 1).unwrap(), vec![(0, 2)]);
        assert!(plan_rounds(&[1, 2, 3], 2, 4).is_err());
        assert_eq!(plan_rounds(&[], 2, 4).unwrap(), vec![]);
        // Every round's distinct experts fit, the rounds cover the rows in order, and
        // each but the last would not fit with the next row.
        let mut rng = 7u64;
        for _ in 0..500 {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let (active, capacity, rows) = (
                1 + u32::try_from((rng >> 60) % 4).unwrap(),
                4 + u32::try_from((rng >> 50) % 8).unwrap(),
                usize::try_from((rng >> 40) % 40).unwrap(),
            );
            let ids: Vec<u32> = (0..rows * active as usize)
                .map(|i| {
                    let mixed = (rng >> (i % 30)) ^ (i as u64).wrapping_mul(2_654_435_761);
                    u32::try_from(mixed % 24).unwrap()
                })
                .collect();
            let rounds = plan_rounds(&ids, active, capacity).unwrap();
            let a = active as usize;
            let distinct = |from: u32, to: u32| {
                ids[from as usize * a..to as usize * a]
                    .iter()
                    .collect::<HashSet<_>>()
                    .len()
            };
            let mut at = 0;
            for (i, &(first, count)) in rounds.iter().enumerate() {
                assert_eq!(first, at);
                assert!(count > 0);
                assert!(distinct(first, first + count) <= capacity as usize);
                if i + 1 < rounds.len() {
                    assert!(distinct(first, first + count + 1) > capacity as usize);
                }
                at += count;
            }
            assert_eq!(at as usize, rows);
        }
    }

    #[test]
    fn fill_splits_the_free_bytes() {
        let per = 18_800_640;
        let gb = 1usize << 30;
        let [a, b] =
            capacities(Capacity::Fill { reserve: 2 * gb }, [62 * gb, 0], true, per).unwrap();
        assert_eq!(a, b);
        assert_eq!(a as usize, 60 * gb / 2 / (per + 8));
        let [a, b] = capacities(
            Capacity::Fill { reserve: gb },
            [10 * gb, 5 * gb],
            false,
            per,
        )
        .unwrap();
        assert!(a > b);
        assert!(capacities(Capacity::Fill { reserve: 2 * gb }, [gb, gb], true, per).is_err());
        assert_eq!(
            capacities(Capacity::Experts([3, 4]), [0, 0], true, per).unwrap(),
            [3, 4]
        );
    }

    /// The released model on one B300 (275,040 MiB), both halves on it with both Engram
    /// tables and the drafter, the backbone's routed experts staged, laid out with no
    /// checkpoint: what is left for the caches after the driver's reserve. The figures
    /// are the module documentation's.
    #[test]
    fn the_released_model_leaves_room_for_thousands_of_experts_on_one_device() {
        use super::super::dry::DryDevice;
        use super::super::support::tests::released_config;
        use super::super::weights::{Form, weight_bytes};
        use eidola_engine::spec::Bucket;
        let c = released_config();
        let e = c.engram.clone().unwrap();
        let map: Vec<u32> = (0..c.vocab_size)
            .map(|i| u32::try_from(i % e.compressed_vocab).unwrap())
            .collect();
        let cfg = DsCudaExecutorConfig {
            block_size: 32,
            num_blocks: 1024,
            num_state_slots: 4,
            max_model_len: 4096,
            buckets: vec![Bucket {
                max_seqs: 1,
                max_tokens: 96,
            }],
            sampleable_vocab_size: u32::try_from(c.vocab_size).unwrap(),
            devices: [0, 0],
            draft_width: 5,
            host_blocks: 0,
            position_tags: false,
            record: false,
            gemm: crate::ds::gemm::GemmForm::Exact,
        };
        let device = || -> Box<dyn Device> { Box::new(DryDevice::new(None)) };
        let x = DsCudaExecutor::build_with(device(), device(), None, &c, Some(map), &cfg, false)
            .unwrap();
        let [enc, dec] = x.memory();
        let shape = check_supported(&c, 5).unwrap();
        let ck = |h| weight_bytes(&c, &shape, h, Form::Checkpoint);
        let experts = ck(Half::Encoder).experts + ck(Half::Decoder).experts;
        // 40 layers of 384 experts of 18,800,640 bytes: 288.8 GB.
        assert_eq!(experts, 40 * 384 * 18_800_640);
        // Neither half holds a routed expert of the backbone; both Engram tables are on
        // the device.
        let tables = 101_377_628_352 + 101_380_404_048;
        assert_eq!(
            x.half(Half::Encoder).w.bytes.engram_tables
                + x.half(Half::Decoder).w.bytes.engram_tables,
            tables
        );
        assert_eq!(x.half(Half::Encoder).w.bytes.experts, 0);
        assert_eq!(x.half(Half::Decoder).w.bytes.experts, 0);
        let rest = enc.total() + dec.total();
        let capacity = 275_040u64 << 20;
        // The driver's context and the kernel images: the node's fixed reserve.
        let reserve = 4u64 << 30;
        let room = capacity - rest - reserve;
        let per = 18_800_640 + 8;
        eprintln!(
            "weights without experts {} + {}, scratch {} + {}, kv {} + {}: {rest} bytes; \
             {room} bytes free for caches: {} experts of 15,360",
            enc.weights,
            dec.weights,
            enc.scratch,
            dec.scratch,
            enc.kv,
            dec.kv,
            room / per
        );
        assert!((221_000_000_000..223_000_000_000).contains(&rest), "{rest}");
        assert!(room / per > 3000, "{}", room / per);
    }
}
