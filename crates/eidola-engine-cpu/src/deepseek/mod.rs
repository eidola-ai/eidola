//! [`DsCpuExecutor`]: the serving core's [`Executor`] for DeepSeek-V4.1 over the model
//! crate's reference numerics, with KV paged in the model's packed formats and the model
//! split across two logical devices. Doctrine: this crate's `AGENTS.md` → DeepSeek-V4.1.

mod device;
mod layer;
mod memory;
pub mod oracle;
mod packed;
mod pool;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use eidola_engine::executor::{Executor, ExecutorError, Maintenance, Slot, StepInput, StepOutput};
use eidola_engine::sampling::{self, Logits, SamplingParams, Stream};
use eidola_engine::spec::{
    AttentionKind, Bucket, Coverage, KvGroupSpec, KvRole, ModelSpec, WindowChain,
};
use eidola_engine_model::deepseek::DsModel;

use device::{Decoder, Encoder, Item, Need, Shape};
pub use device::{StepTransfer, ToDecoder, ToEncoder};
use memory::Memory;
pub use oracle::{DsDenseOracle, DsDenseRun};

/// Executor configuration: memory geometry and the draft width. The model fixes the rest.
#[derive(Clone, Debug, PartialEq)]
pub struct DsCpuExecutorConfig {
    /// Positions per KV block (every group); a multiple of the pooling ratio.
    pub block_size: u32,
    /// Physical blocks per group, including the null block 0.
    pub num_blocks: u32,
    /// Host-tier blocks per group, including the null block 0; 0 for no host tier.
    pub host_blocks: u32,
    /// Per-sequence state rows.
    pub num_state_slots: u32,
    /// Longest sequence.
    pub max_model_len: u32,
    /// Captured step shapes, ascending.
    pub buckets: Vec<Bucket>,
    /// Drafted tokens per speculative step, at most the drafter's slots; 0 disables
    /// drafting and the drafter's KV group.
    pub draft_width: u32,
    /// Token ids the model may emit; the head's rows past it take no part in sampling.
    pub sampleable_vocab_size: u32,
    /// Testing aid: keep a [`DsRowRecord`] of every row.
    pub record: bool,
}

impl DsCpuExecutorConfig {
    /// A configuration drafting with every slot of the model's drafter.
    pub fn for_model(model: &DsModel, sampleable_vocab_size: u32) -> Self {
        let c = model.config();
        Self {
            block_size: 16,
            num_blocks: 256,
            host_blocks: 0,
            num_state_slots: 16,
            max_model_len: c.max_positions.unwrap_or(4096) as u32,
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
            draft_width: match (&c.drafter, &model.weights.drafter) {
                (Some(d), Some(_)) => d.slots as u32,
                _ => 0,
            },
            sampleable_vocab_size,
            record: false,
        }
    }
}

/// Everything one row of one step computed, for checking against the dense oracle.
#[derive(Clone, Debug, PartialEq)]
pub struct DsRowRecord {
    pub slot: Slot,
    /// Positions already in KV before the step.
    pub context_len: u32,
    /// Host tokens in the row.
    pub num_tokens: u32,
    /// Where the window chain's top layer started.
    pub chain_start: u32,
    /// Tokens at positions `0 ..= p + k`: the prefix read back through the block tables,
    /// the host tokens, then the drafts.
    pub tokens: Vec<u32>,
    /// Target logits by position: the last host position of a sampling row, and every
    /// drafted position.
    pub target_logits: Vec<(u32, Vec<f32>)>,
    /// The drafter's logits per slot, each given the token before it in the chain.
    pub draft_logits: Vec<Vec<f32>>,
    /// Drafted tokens.
    pub drafts: Vec<u32>,
    /// Tokens the row produced.
    pub produced: Vec<u32>,
    pub sampling: SamplingParams,
}

/// A row of a step in flight.
struct Row {
    slot: Slot,
    c: u32,
    p: u32,
    k: u32,
    sample: bool,
    params: SamplingParams,
    chain_start: u32,
    /// The ids before `c` the step was given (those that exist), then the tokens at
    /// `c ..`: host tokens, then drafts.
    known: Vec<u32>,
    history: usize,
    drafts: Vec<u32>,
    q_rows: Vec<Vec<f64>>,
    draft_logits: Vec<Vec<f32>>,
    target_logits: HashMap<u32, Vec<f32>>,
    produced: Vec<u32>,
}

impl Row {
    fn token(&self, pos: u32) -> u32 {
        self.known[self.history + (pos - self.c) as usize]
    }

    /// The token at `pos` and up to `n - 1` before it, oldest first.
    fn lookback(&self, pos: u32, n: usize) -> &[u32] {
        let at = self.history + (pos - self.c) as usize;
        &self.known[(at + 1).saturating_sub(n)..=at]
    }
}

/// The CPU reference executor for DeepSeek-V4.1. See the module documentation.
pub struct DsCpuExecutor {
    spec: ModelSpec,
    cfg: DsCpuExecutorConfig,
    shape: Shape,
    encoder: Encoder,
    decoder: Decoder,
    records: Mutex<Vec<DsRowRecord>>,
    transfers: Mutex<Vec<StepTransfer>>,
    zero_log: Vec<(u64, u32, u32)>,
    crypt_log: Vec<(u64, u32, u32, bool)>,
    transfer_log: Vec<(u64, u32, u32, u32, bool)>,
    host_zero_log: Vec<(u64, u32, u32)>,
    steps: u64,
}

impl std::fmt::Debug for DsCpuExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DsCpuExecutor")
            .field("spec", &self.spec)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

impl DsCpuExecutor {
    /// An executor over `model`. Panics if the model's shape is not one this executor
    /// runs (see `AGENTS.md` → DeepSeek-V4.1 → Shape) or the configuration does not fit.
    pub fn new(model: Arc<DsModel>, cfg: DsCpuExecutorConfig) -> Self {
        let shape = Shape::of(&model, cfg.draft_width);
        let c = model.config();
        let window = c.attention.window as u32;
        let hd = c.attention.head_dim as u32;
        let group =
            |name: &str, role, attention, layers: usize, ratio, device, coverage| KvGroupSpec {
                name: name.into(),
                role,
                attention,
                num_layers: layers as u32,
                num_kv_heads: 1,
                head_dim_qk: hd,
                head_dim_v: hd,
                num_blocks: cfg.num_blocks,
                host_blocks: cfg.host_blocks,
                ratio,
                device,
                coverage,
            };
        let (target, all) = (KvRole::Target, Coverage::All);
        let sliding = |window| AttentionKind::Sliding { window };
        let mut kv_groups = vec![
            group(
                "encoder-bank",
                target,
                AttentionKind::Full,
                shape.sources.len(),
                shape.ratio,
                0,
                all,
            ),
            group(
                "encoder-window",
                target,
                sliding(window),
                shape.split,
                1,
                0,
                all,
            ),
            group("decoder-bank", target, AttentionKind::Full, 1, 1, 1, all),
            group(
                "decoder-window",
                target,
                sliding(window),
                shape.depth(),
                1,
                1,
                Coverage::Chain,
            ),
        ];
        if cfg.draft_width > 0 {
            // A drafting row reads the `window` positions before its own.
            kv_groups.push(group(
                "drafter",
                KvRole::Drafter,
                sliding(window + 1),
                shape.drafter_blocks,
                1,
                1,
                Coverage::Chain,
            ));
        }
        let spec = ModelSpec {
            vocab_size: c.vocab_size as u32,
            sampleable_vocab_size: cfg.sampleable_vocab_size,
            block_size: cfg.block_size,
            max_model_len: cfg.max_model_len,
            kv_groups,
            max_draft_tokens: cfg.draft_width,
            num_state_slots: cfg.num_state_slots,
            buckets: cfg.buckets.clone(),
            window_chain: Some(WindowChain {
                depth: shape.depth() as u32,
                window,
            }),
            token_history: shape.history as u32,
        };
        spec.validate().expect("valid model spec");
        let encoder = Encoder::new(model.clone(), &shape, &spec, &cfg);
        let decoder = Decoder::new(model, &shape, &spec, &cfg);
        Self {
            records: Mutex::new(Vec::new()),
            transfers: Mutex::new(Vec::new()),
            zero_log: Vec::new(),
            crypt_log: Vec::new(),
            transfer_log: Vec::new(),
            host_zero_log: Vec::new(),
            steps: 0,
            spec,
            cfg,
            shape,
            encoder,
            decoder,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &DsCpuExecutorConfig {
        &self.cfg
    }

    /// Steps executed.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Layer-positions the window chain has computed.
    pub fn chain_layer_positions(&self) -> u64 {
        self.decoder.chain_work
    }

    /// The first decoder layer: the layers before it run on device 0.
    pub fn split(&self) -> usize {
        self.shape.split
    }

    fn memory(&self, group: u32) -> (&Memory, usize) {
        [&self.encoder.mem, &self.decoder.mem]
            .into_iter()
            .find_map(|m| m.local(group).map(|g| (m, g)))
            .unwrap_or_else(|| panic!("no group {group}"))
    }

    fn memory_mut(&mut self, group: u32) -> (&mut Memory, usize) {
        match self.encoder.mem.local(group) {
            Some(g) => (&mut self.encoder.mem, g),
            None => {
                let g = self
                    .decoder
                    .mem
                    .local(group)
                    .unwrap_or_else(|| panic!("no group {group}"));
                (&mut self.decoder.mem, g)
            }
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

    /// Every executed demote (`true`) or promote as `(step, group, block, host, demote)`.
    pub fn transfer_log(&self) -> &[(u64, u32, u32, u32, bool)] {
        &self.transfer_log
    }

    /// Every executed host zero as `(step, group, host)`.
    pub fn host_zero_log(&self) -> &[(u64, u32, u32)] {
        &self.host_zero_log
    }

    /// Whether every byte of a block (packed KV and tags) is zero.
    pub fn block_is_zero(&self, group: u32, block: u32) -> bool {
        let (m, g) = self.memory(group);
        m.groups[g].pool.block_is_zero(block)
    }

    /// Whether a block is held encrypted.
    pub fn block_is_encrypted(&self, group: u32, block: u32) -> bool {
        let (m, g) = self.memory(group);
        m.groups[g].pool.is_encrypted(block)
    }

    /// A block's stored image as words (the cells' bytes, then the tags), as the
    /// encryption covers it: plaintext or ciphertext, whichever the block holds.
    pub fn block_image(&self, group: u32, block: u32) -> Vec<u32> {
        let (m, g) = self.memory(group);
        m.groups[g].pool.image(block)
    }

    /// Bytes of packed KV in one block of a group.
    pub fn block_bytes(&self, group: u32) -> usize {
        let (m, g) = self.memory(group);
        m.groups[g].pool.block_bytes()
    }

    fn host(&self, group: u32) -> &pool::BytePool {
        let (m, g) = self.memory(group);
        m.groups[g].host.as_ref().expect("a host tier")
    }

    /// A host-tier block's stored image, in [`DsCpuExecutor::block_image`]'s order.
    pub fn host_block_image(&self, group: u32, host: u32) -> Vec<u32> {
        self.host(group).image(host)
    }

    /// Whether every byte of a host-tier block is zero.
    pub fn host_block_is_zero(&self, group: u32, host: u32) -> bool {
        self.host(group).block_is_zero(host)
    }

    /// Whether a host-tier block holds ciphertext.
    pub fn host_block_is_encrypted(&self, group: u32, host: u32) -> bool {
        self.host(group).is_encrypted(host)
    }

    /// Whether a slot's per-sequence model state (the open entries of the pooled
    /// sources) is scrubbed.
    pub fn slot_state_is_zero(&self, slot: Slot) -> bool {
        self.encoder.open.is_zero(slot)
    }

    /// Records kept since the last call (with [`DsCpuExecutorConfig::record`]).
    pub fn take_records(&self) -> Vec<DsRowRecord> {
        std::mem::take(&mut *self.records.lock().expect("records lock"))
    }

    /// What crossed between the devices in each step since the last call (with
    /// [`DsCpuExecutorConfig::record`]).
    pub fn transfers(&self) -> Vec<StepTransfer> {
        std::mem::take(&mut *self.transfers.lock().expect("transfers lock"))
    }

    fn maintain(&mut self, step: &StepInput) {
        for m in &step.maintenance {
            match *m {
                Maintenance::Zero { group, block } => {
                    let (mem, g) = self.memory_mut(group);
                    mem.groups[g].pool.zero(block);
                    self.zero_log.push((self.steps, group, block));
                }
                Maintenance::Copy { group, src, dst } => {
                    let (mem, g) = self.memory_mut(group);
                    mem.groups[g].pool.copy(src, dst);
                }
                Maintenance::ResetSlot { slot } => {
                    self.encoder.mem.reset_slot(slot);
                    self.decoder.mem.reset_slot(slot);
                    self.encoder.open.scrub(slot);
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
                    let key = step
                        .kv_keys
                        .get(key as usize)
                        .unwrap_or_else(|| panic!("maintenance names key {key}, not in the step"));
                    let (mem, g) = self.memory_mut(group);
                    mem.groups[g].pool.crypt(block, key, &nonce, encrypt);
                    self.crypt_log.push((self.steps, group, block, encrypt));
                }
                Maintenance::Demote { group, block, host } => {
                    let (mem, g) = self.memory_mut(group);
                    let gm = &mut mem.groups[g];
                    assert!(
                        gm.pool.is_encrypted(block),
                        "group {group} block {block}: demote of a plaintext block"
                    );
                    let tier = gm.host.as_mut().expect("a host tier");
                    pool::BytePool::transfer(&gm.pool, block, tier, host);
                    self.transfer_log
                        .push((self.steps, group, block, host, true));
                }
                Maintenance::Promote { group, host, block } => {
                    let (mem, g) = self.memory_mut(group);
                    let gm = &mut mem.groups[g];
                    let tier = gm.host.as_ref().expect("a host tier");
                    assert!(
                        tier.is_encrypted(host),
                        "group {group} host block {host}: promote of a block holding no ciphertext"
                    );
                    assert!(
                        gm.pool.block_is_zero(block) && !gm.pool.is_encrypted(block),
                        "group {group} block {block}: promote into a block that is not zero"
                    );
                    pool::BytePool::transfer(tier, host, &mut gm.pool, block);
                    self.transfer_log
                        .push((self.steps, group, block, host, false));
                }
                Maintenance::ZeroHost { group, host } => {
                    let (mem, g) = self.memory_mut(group);
                    mem.groups[g].host.as_mut().expect("a host tier").zero(host);
                    self.host_zero_log.push((self.steps, group, host));
                }
            }
        }
        for u in &step.table_updates {
            let (mem, g) = self.memory_mut(u.group);
            mem.set_table(g, u.slot, u.index, u.block);
        }
    }
}

impl Executor for DsCpuExecutor {
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
        self.maintain(step);

        let h = self.shape.history;
        let (vocab, sampleable) = (self.spec.vocab_size, self.spec.sampleable_vocab_size);
        let mut rows: Vec<Row> = Vec::with_capacity(step.seqs.len());
        for (ri, e) in step.seqs.iter().enumerate() {
            assert!(e.num_tokens >= 1, "a row needs at least one host token");
            let start = e.token_start as usize;
            let end = start + e.num_tokens as usize;
            for (i, &pos) in step.positions[start..end].iter().enumerate() {
                assert_eq!(
                    pos,
                    e.context_len + i as u32,
                    "positions disagree with context_len"
                );
            }
            let p = e.context_len + e.num_tokens - 1;
            let k = if e.sample { e.num_drafts } else { 0 };
            assert!(
                k <= self.cfg.draft_width,
                "{k} drafts requested; the executor drafts {}",
                self.cfg.draft_width
            );
            assert!(
                p > 0 || k == 0,
                "slot {}: {k} drafts requested at position 0",
                e.slot
            );
            let chain_start = *step
                .aux
                .chain_start
                .get(ri)
                .expect("a row of a model with a window chain carries its chain_start");
            assert!(
                !e.sample || chain_start <= p,
                "slot {}: sampling at {p}, where the chain's top layer does not run",
                e.slot
            );
            assert!(
                k == 0 || chain_start == e.context_len,
                "slot {}: a drafting row runs every chain layer at every position",
                e.slot
            );
            let given = step
                .aux
                .history_ids
                .get(ri * h..(ri + 1) * h)
                .expect("a row carries its token history");
            let history = h.min(e.context_len as usize);
            let mut known = given[h - history..].to_vec();
            known.extend_from_slice(&step.token_ids[start..end]);
            rows.push(Row {
                slot: e.slot,
                c: e.context_len,
                p,
                k,
                sample: e.sample,
                params: e.sampling,
                chain_start,
                known,
                history,
                drafts: Vec::new(),
                q_rows: Vec::new(),
                draft_logits: Vec::new(),
                target_logits: HashMap::new(),
                produced: Vec::new(),
            });
        }

        // 1. Drafting, on device 1, from KV written by earlier steps: the drafter's
        //    context before `p` and the host token at `p`. The slots' logits form a
        //    chain: each is conditioned on the token drafted before it.
        let mut transfer = StepTransfer::default();
        for row in rows.iter_mut().filter(|r| r.k > 0) {
            let core = self.decoder.draft(row.slot, row.p, row.token(row.p))?;
            let mut previous = row.token(row.p);
            for i in 0..row.k as usize {
                let logits = self.decoder.draft_logits(&core, i, previous);
                assert_eq!(logits.len(), vocab as usize);
                let view = Logits::new(&logits, sampleable);
                let pos = row.p as u64 + 1 + i as u64;
                let (d, q) = if row.params.is_greedy() {
                    let d = sampling::argmax(view);
                    let mut q = vec![0.0; view.len()];
                    q[d as usize] = 1.0;
                    (d, q)
                } else {
                    let q = sampling::processed_probs(view, &row.params);
                    let u = sampling::uniform(row.params.seed(), pos, Stream::Draft);
                    (sampling::sample_from(&q, u), q)
                };
                row.drafts.push(d);
                row.known.push(d);
                row.q_rows.push(q);
                row.draft_logits.push(logits);
                previous = d;
            }
            transfer.drafts.push((row.slot, row.drafts.clone()));
        }

        // 2. The positions of the step: every host token, then the drafts.
        let depth = self.shape.depth() as u32;
        let max_ngram = h + 1;
        let (mut items, mut needs) = (Vec::new(), Vec::new());
        for (ri, row) in rows.iter().enumerate() {
            let bottom = self.spec.chain_layer_start(row.chain_start, 0);
            for pos in row.c..=row.p + row.k {
                items.push(Item {
                    row: ri,
                    slot: row.slot,
                    pos,
                    token: row.token(pos),
                    lookback: row.lookback(pos, max_ngram).to_vec(),
                });
                needs.push(Need {
                    boundary: pos >= bottom,
                    taps: pos >= row.chain_start,
                });
            }
        }

        // 3. Device 1 gathers the Engram rows of the tables it holds; device 0 runs the
        //    encoder and the first decoder layer's bank stub; the typed crossing goes to
        //    device 1, which runs the chain.
        let gathered = self.decoder.gather(&items)?;
        transfer.note_to_encoder(&items, &gathered);
        let crossing = self.encoder.forward(&items, &needs, &gathered)?;
        crossing.check(&items, &needs);
        transfer.note_to_decoder(&crossing);
        let starts: Vec<Vec<u32>> = rows
            .iter()
            .map(|r| {
                (0..depth)
                    .map(|j| self.spec.chain_layer_start(r.chain_start, j))
                    .collect()
            })
            .collect();
        let top: Vec<bool> = items
            .iter()
            .map(|it| rows[it.row].sample && it.pos >= rows[it.row].p)
            .collect();
        let logits = self
            .decoder
            .forward(&items, &needs, &starts, &top, crossing)?;
        for (it, l) in items.iter().zip(logits) {
            if let Some(l) = l {
                rows[it.row].target_logits.insert(it.pos, l);
            }
        }

        // 4. Sampling and chain acceptance.
        for row in &mut rows {
            if !row.sample {
                continue;
            }
            let params = row.params;
            row.produced = if row.k == 0 {
                vec![sampling::sample(
                    self.spec.logits(&row.target_logits[&row.p]),
                    &params,
                    row.p as u64 + 1,
                )]
            } else {
                let p_rows: Vec<Vec<f64>> = (row.p..=row.p + row.k)
                    .map(|pos| {
                        sampling::processed_probs(
                            self.spec.logits(&row.target_logits[&pos]),
                            &params,
                        )
                    })
                    .collect();
                sampling::chain_accept(&p_rows, &row.q_rows, &row.drafts, &params, row.p as u64 + 1)
            };
        }

        let stride = self.spec.max_draft_tokens + 1;
        let mut out = StepOutput {
            tokens: vec![0; rows.len() * stride as usize],
            stride,
            num_tokens: vec![0; rows.len()],
            logits: step.return_logits.then(Vec::new),
        };
        for (i, row) in rows.iter().enumerate() {
            let base = i * stride as usize;
            out.tokens[base..base + row.produced.len()].copy_from_slice(&row.produced);
            out.num_tokens[i] = row.produced.len() as u32;
            if let Some(l) = out.logits.as_mut() {
                l.push(
                    (0..row.produced.len() as u32)
                        .map(|j| row.target_logits[&(row.p + j)].clone())
                        .collect(),
                );
            }
        }
        if self.cfg.record {
            self.transfers
                .get_mut()
                .expect("transfers lock")
                .push(transfer);
            for row in rows {
                let mut tokens = self.decoder.prefix_tokens(row.slot, row.c);
                tokens.extend(&row.known[row.history..]);
                let mut target_logits: Vec<(u32, Vec<f32>)> =
                    row.target_logits.into_iter().collect();
                target_logits.sort_by_key(|(p, _)| *p);
                self.records
                    .get_mut()
                    .expect("records lock")
                    .push(DsRowRecord {
                        slot: row.slot,
                        context_len: row.c,
                        num_tokens: row.p + 1 - row.c,
                        chain_start: row.chain_start,
                        tokens,
                        target_logits,
                        draft_logits: row.draft_logits,
                        drafts: row.drafts,
                        produced: row.produced,
                        sampling: row.params,
                    });
            }
        }
        Ok(out)
    }
}
