//! The mock's second model: a two-device encoder/decoder with a window chain, pooled and
//! per-position global banks read through a top-k selector, a token-history lookup, and
//! a block drafter with a Markov dependency (the DeepSeek-V4.1 shape, in hashes).
//!
//! Per position `x` holding token `t`:
//!
//! - **Encoder** (device 0). `h0 = mix(t, x, the token_history ids before x)`. Its window
//!   KV at `x` is a function of `h0`. Bank entry `j` of the pooled group folds `h0` over
//!   the `ratio` positions it covers and is written at the last of them; the values of
//!   an open entry's earlier positions wait in per-slot state. The encoder's output `e`
//!   folds its window and the top [`TOP_K`] visible bank entries by a score against `h0`.
//! - **Decoder** (device 1). The per-position bank holds a function of `e`. Chain layer
//!   `j`'s input is `e` (bottom) or layer `j - 1`'s output; its window KV at `x` is a
//!   function of that input, and its output folds its own window, its input and, at the
//!   bottom and middle layers, the top [`TOP_K`] entries of the per-position bank. The
//!   top layer's output seeds the logits.
//! - **Drafter** (device 1). Its context KV at `x` is a function of the inputs of the top
//!   [`TAPS`] chain layers at `x`. A row whose last host position is `p` drafts from the
//!   context at the positions before `p`, the host token at `p`, the slot index, and the
//!   token drafted for the slot before.
//!
//! [`Reference`] runs every layer at every position. The executor runs the chain by the
//! row rule of [`crate::executor`] and fails on a read of any cell that was never
//! written: a position the chain skipped, at the layer that skipped it.

use super::{MockConfig, MockExecutor, logits_from_state};
use crate::executor::{ExecutorError, SeqEntry, StepInput};
use crate::sampling::{self, SamplingParams, Stream, mix64};
use crate::spec::{AttentionKind, Coverage, KvRole, ModelSpec};

/// Bank entries a query attends to.
pub const TOP_K: usize = 3;
/// Top chain layers whose inputs the drafter's context is projected from.
pub const TAPS: u32 = 3;

/// Group indices of the chain model, fixed by [`super::deepseek_like_spec`].
pub(super) const ENC_BANK: u32 = 0;
pub(super) const ENC_WINDOW: u32 = 1;
pub(super) const DEC_BANK: u32 = 2;
pub(super) const DEC_WINDOW: u32 = 3;
pub(super) const DRAFTER: u32 = 4;

/// The shape numbers of a chain spec.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shape {
    ratio: u32,
    enc_window: u32,
    depth: u32,
    window: u32,
    drafter_window: u32,
    history: u32,
}

impl Shape {
    /// Reads the shape from a spec laid out as [`super::deepseek_like_spec`] does.
    pub(super) fn of(spec: &ModelSpec) -> Self {
        let chain = spec.window_chain.expect("a chain spec");
        let window = |g: u32| match spec.kv_groups[g as usize].attention {
            AttentionKind::Sliding { window } => window,
            AttentionKind::Full => panic!("group {g} of a chain spec slides"),
        };
        let g = &spec.kv_groups;
        assert_eq!(g.len(), 5, "a chain spec has five groups");
        assert!(g[ENC_BANK as usize].attention == AttentionKind::Full);
        assert!(g[DEC_BANK as usize].attention == AttentionKind::Full);
        assert!(g[DEC_BANK as usize].ratio == 1);
        assert!(g[DEC_WINDOW as usize].coverage == Coverage::Chain);
        assert!(g[DRAFTER as usize].coverage == Coverage::Chain);
        assert!(g[DRAFTER as usize].role == KvRole::Drafter);
        assert!(g[ENC_BANK as usize].device == g[ENC_WINDOW as usize].device);
        assert!(g[ENC_BANK as usize].device != g[DEC_WINDOW as usize].device);
        Self {
            ratio: g[ENC_BANK as usize].ratio,
            enc_window: window(ENC_WINDOW),
            depth: chain.depth,
            window: chain.window,
            drafter_window: window(DRAFTER),
            history: spec.token_history,
        }
    }

    fn uses_bank(&self, layer: u32) -> bool {
        layer == 0 || layer == self.depth / 2
    }

    fn taps(&self) -> u32 {
        TAPS.min(self.depth)
    }
}

fn tag(cfg: &MockConfig, kind: u64, a: u64, b: u64) -> u64 {
    mix64(cfg.model_seed ^ mix64(kind.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (a << 32) ^ b))
}

/// `history` holds the ids before `pos`, oldest first.
fn h0(cfg: &MockConfig, token: u32, pos: u32, history: &[u32]) -> u64 {
    let mut acc = tag(cfg, 1, token as u64, pos as u64);
    for (i, &t) in history.iter().rev().enumerate() {
        acc = mix64(acc ^ tag(cfg, 2, t as u64, i as u64));
    }
    acc
}

fn enc_kv(cfg: &MockConfig, h0: u64) -> u64 {
    mix64(h0 ^ tag(cfg, 3, 0, 0)) | 1
}

fn pool(cfg: &MockConfig, parts: &[u64]) -> u64 {
    let mut acc = tag(cfg, 4, 0, 0);
    for &p in parts {
        acc = mix64(acc ^ p);
    }
    acc | 1
}

fn dec_bank_kv(cfg: &MockConfig, e: u64) -> u64 {
    mix64(e ^ tag(cfg, 5, 0, 0)) | 1
}

fn chain_kv(cfg: &MockConfig, input: u64, layer: u32, pos: u32) -> u64 {
    mix64(input ^ tag(cfg, 6, layer as u64, pos as u64)) | 1
}

fn drafter_kv(cfg: &MockConfig, inputs: &[u64], pos: u32) -> u64 {
    let mut acc = tag(cfg, 7, 0, pos as u64);
    for &i in inputs {
        acc = mix64(acc ^ i);
    }
    acc | 1
}

fn fold(seed: u64, values: impl Iterator<Item = u64>) -> u64 {
    let mut acc = seed;
    for v in values {
        acc = mix64(acc ^ v);
    }
    acc
}

/// The top-k selector: the [`TOP_K`] entries scoring highest against `query` (ties to
/// the lower index), folded in index order.
fn select(query: u64, entries: &[u64]) -> u64 {
    let mut best: Vec<(u64, usize)> = entries
        .iter()
        .enumerate()
        .map(|(i, &v)| (mix64(query ^ v), i))
        .collect();
    best.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    best.truncate(TOP_K);
    best.sort_unstable_by_key(|&(_, i)| i);
    fold(0x5e1ec7, best.iter().map(|&(_, i)| entries[i]))
}

fn encoder_out(h0: u64, window: u64, selected: u64) -> u64 {
    mix64(h0 ^ mix64(window ^ 0xe1) ^ mix64(selected ^ 0xe2))
}

fn layer_out(input: u64, window: u64, selected: Option<u64>) -> u64 {
    let s = selected.map_or(0, |s| mix64(s ^ 0xd2));
    mix64(input ^ mix64(window ^ 0xd1) ^ s)
}

fn draft_noise(context: u64, slot: u32, prev: u32) -> u64 {
    mix64(context ^ mix64(((slot as u64) << 32) ^ prev as u64 ^ 0xd4af7))
}

fn first_visible(window: u32, q: u32) -> u32 {
    (q + 1).saturating_sub(window)
}

/// The dense oracle: every layer at every position, extended one token at a time.
#[derive(Debug)]
pub struct Reference {
    shape: Shape,
    cfg: MockConfig,
    vocab: u32,
    tokens: Vec<u32>,
    h0: Vec<u64>,
    enc_bank: Vec<u64>,
    dec_bank: Vec<u64>,
    /// `[layer][pos]`.
    kv: Vec<Vec<u64>>,
    drafter: Vec<u64>,
    top_out: Vec<u64>,
}

impl Reference {
    /// An empty sequence of the model `spec` and `cfg` describe.
    pub fn new(spec: &ModelSpec, cfg: &MockConfig) -> Self {
        let shape = Shape::of(spec);
        Self {
            shape,
            cfg: *cfg,
            vocab: spec.vocab_size,
            tokens: Vec::new(),
            h0: Vec::new(),
            enc_bank: Vec::new(),
            dec_bank: Vec::new(),
            kv: vec![Vec::new(); shape.depth as usize],
            drafter: Vec::new(),
            top_out: Vec::new(),
        }
    }

    /// Appends a token and runs every layer at its position.
    pub fn push(&mut self, token: u32) {
        let (s, cfg) = (self.shape, self.cfg);
        let x = self.tokens.len() as u32;
        self.tokens.push(token);
        let lo = x.saturating_sub(s.history) as usize;
        let h = h0(&cfg, token, x, &self.tokens[lo..x as usize]);
        self.h0.push(h);
        if (x + 1).is_multiple_of(s.ratio) {
            let first = (x + 1 - s.ratio) as usize;
            self.enc_bank.push(pool(&cfg, &self.h0[first..]));
        }
        let window = fold(
            0,
            (first_visible(s.enc_window, x)..=x).map(|p| enc_kv(&cfg, self.h0[p as usize])),
        );
        let e = encoder_out(h, window, select(h, &self.enc_bank));
        self.dec_bank.push(dec_bank_kv(&cfg, e));
        let mut input = e;
        let mut inputs = Vec::with_capacity(s.depth as usize);
        for j in 0..s.depth {
            inputs.push(input);
            self.kv[j as usize].push(chain_kv(&cfg, input, j, x));
            let lo = first_visible(s.window, x) as usize;
            let window = fold(0, self.kv[j as usize][lo..].iter().copied());
            let selected = s.uses_bank(j).then(|| select(input, &self.dec_bank));
            input = layer_out(input, window, selected);
        }
        self.top_out.push(input);
        let taps = &inputs[(s.depth - s.taps()) as usize..];
        self.drafter.push(drafter_kv(&cfg, taps, x));
    }

    /// The target logits for the token after position `q`.
    pub fn logits(&self, q: u32) -> Vec<f32> {
        logits_from_state(&self.cfg, self.vocab, self.top_out[q as usize])
    }

    /// Positions pushed.
    pub fn len(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// Whether nothing was pushed.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// What one position's forward leaves for the row.
struct Computed {
    /// The top layer's output, where it was asked for.
    top_out: Option<u64>,
}

impl MockExecutor {
    fn device_of(&self, group: u32) -> u32 {
        self.spec.kv_groups[group as usize].device
    }

    /// Runs one position of a row whose chain top starts at `chain_start`: the encoder
    /// always, each chain layer where the row rule has it compute, the top layer's
    /// output when `want_out`. `history` holds the ids before `x`, oldest first.
    fn chain_position(
        &mut self,
        slot: u32,
        x: u32,
        token: u32,
        history: &[u32],
        chain_start: u32,
        want_out: bool,
    ) -> Result<Computed, ExecutorError> {
        let s = self.chain.expect("chain model");
        let cfg = self.cfg;

        // Encoder: device 0 only; `e` is what crosses.
        let dev = self.device_of(ENC_WINDOW);
        let h = h0(&cfg, token, x, history);
        self.put(slot, ENC_WINDOW, 0, x, enc_kv(&cfg, h), dev)?;
        let pairs = &mut self.pairs[slot as usize];
        pairs.retain(|&(p, _)| p != x && p + 64 > x);
        pairs.push((x, h));
        if (x + 1).is_multiple_of(s.ratio) {
            let mut parts = Vec::with_capacity(s.ratio as usize);
            for p in x + 1 - s.ratio..=x {
                let Some(&(_, v)) = self.pairs[slot as usize].iter().find(|e| e.0 == p) else {
                    return Err(ExecutorError(format!(
                        "slot {slot} position {x}: the open entry's state at {p} is lost"
                    )));
                };
                parts.push(v);
            }
            self.put(slot, ENC_BANK, 0, x, pool(&cfg, &parts), dev)?;
        }
        let mut window = 0;
        for p in first_visible(s.enc_window, x)..=x {
            window = mix64(window ^ self.get(slot, ENC_WINDOW, 0, p, dev)?);
        }
        let entries = self.bank(slot, ENC_BANK, x, dev)?;
        let e = encoder_out(h, window, select(h, &entries));

        // Decoder: device 1 only.
        let dev = self.device_of(DEC_WINDOW);
        self.put(slot, DEC_BANK, 0, x, dec_bank_kv(&cfg, e), dev)?;
        let mut input = e;
        let mut inputs = Vec::with_capacity(s.depth as usize);
        let mut top_out = None;
        for j in 0..s.depth {
            if x < self.spec.chain_layer_start(chain_start, j) {
                break;
            }
            inputs.push(input);
            self.put(slot, DEC_WINDOW, j, x, chain_kv(&cfg, input, j, x), dev)?;
            self.chain_writes += 1;
            let consumed = if j + 1 < s.depth {
                x >= self.spec.chain_layer_start(chain_start, j + 1)
            } else {
                want_out
            };
            if !consumed {
                break;
            }
            let mut window = 0;
            for p in first_visible(s.window, x)..=x {
                window = mix64(window ^ self.get(slot, DEC_WINDOW, j, p, dev)?);
            }
            let selected = if s.uses_bank(j) {
                Some(select(input, &self.bank(slot, DEC_BANK, x, dev)?))
            } else {
                None
            };
            input = layer_out(input, window, selected);
            if j + 1 == s.depth {
                top_out = Some(input);
            }
        }
        if x >= chain_start {
            let taps = &inputs[(s.depth - s.taps()) as usize..];
            self.put(slot, DRAFTER, 0, x, drafter_kv(&cfg, taps, x), dev)?;
        }
        if want_out && top_out.is_none() {
            return Err(ExecutorError(format!(
                "slot {slot} position {x}: sampling where the chain's top layer does not run"
            )));
        }
        Ok(Computed { top_out })
    }

    /// Every entry of a full-attention bank visible to a query at `q`.
    fn bank(&self, slot: u32, group: u32, q: u32, dev: u32) -> Result<Vec<u64>, ExecutorError> {
        let gs = &self.spec.kv_groups[group as usize];
        let ratio = gs.ratio;
        (0..gs.visible_entries(q))
            .map(|j| self.get(slot, group, 0, j * ratio + ratio - 1, dev))
            .collect()
    }

    /// The drafter's context for a row whose last host position is `p`: its window
    /// before `p`.
    fn drafter_context(&self, slot: u32, p: u32) -> Result<u64, ExecutorError> {
        let s = self.chain.expect("chain model");
        let dev = self.device_of(DRAFTER);
        let mut acc = 0;
        for x in first_visible(s.drafter_window, p)..p {
            acc = mix64(acc ^ self.get(slot, DRAFTER, 0, x, dev)?);
        }
        Ok(acc)
    }

    /// One row of the chain model. Returns the produced tokens and their logits rows.
    pub(super) fn chain_row(
        &mut self,
        step: &StepInput,
        row: usize,
        e: &SeqEntry,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>), ExecutorError> {
        let s = self.chain.expect("chain model");
        let h = s.history as usize;
        let Some(&chain_start) = step.aux.chain_start.get(row) else {
            return Err(ExecutorError("a chain row needs its chain_start".into()));
        };
        let Some(history) = step.aux.history_ids.get(row * h..(row + 1) * h) else {
            return Err(ExecutorError("a row needs its token history".into()));
        };
        if e.sample && e.num_tokens == 1 && chain_start != e.context_len {
            return Err(ExecutorError(
                "a decode row runs every chain layer at every position".into(),
            ));
        }
        // The ids from `context_len - h` on: history, host tokens, then drafts.
        let mut known = history.to_vec();
        known.extend_from_slice(
            &step.token_ids[e.token_start as usize..(e.token_start + e.num_tokens) as usize],
        );
        let before = |known: &[u32], x: u32| {
            let i = (x - e.context_len) as usize + h;
            known[i - h.min(x as usize)..i].to_vec()
        };
        let last = e.context_len + e.num_tokens - 1;
        let mut top = None;
        for i in 0..e.num_tokens {
            let x = e.context_len + i;
            if step.positions[(e.token_start + i) as usize] != x {
                return Err(ExecutorError("positions disagree with context_len".into()));
            }
            let token = known[h + i as usize];
            let hist = before(&known, x);
            let want = e.sample && x == last;
            top = self
                .chain_position(e.slot, x, token, &hist, chain_start, want)?
                .top_out;
        }
        if !e.sample {
            return Ok((Vec::new(), Vec::new()));
        }
        if last == 0 && e.num_drafts > 0 {
            return Err(ExecutorError(
                "drafts requested at position 0, where none can exist".into(),
            ));
        }
        let params = e.sampling;
        let vocab = self.spec.vocab_size;
        let logits = |ex: &Self, state: u64| logits_from_state(&ex.cfg, vocab, state);
        let top = top.expect("a sampling row's top output");
        if e.num_drafts == 0 {
            let l = logits(self, top);
            let t = sampling::sample(self.spec.logits(&l), &params, last as u64 + 1);
            return Ok((vec![t], vec![l]));
        }
        let k = e.num_drafts as usize;
        let context = self.drafter_context(e.slot, last)?;
        let mut states = vec![top];
        let (mut drafts, mut q_rows) = (Vec::with_capacity(k), Vec::with_capacity(k));
        let mut prev = known[h + e.num_tokens as usize - 1];
        for j in 0..k {
            let q = last + j as u32;
            let l = logits(self, states[j]);
            let p = sampling::processed_probs(self.spec.logits(&l), &params);
            let noise = draft_noise(context, j as u32, prev);
            let qd = self.mix_draft(noise, &p, &params);
            let d = draw_draft(&qd, &params, q);
            known.push(d);
            let hist = before(&known, q + 1);
            let c = self.chain_position(e.slot, q + 1, d, &hist, chain_start, true)?;
            states.push(c.top_out.expect("a verified position's top output"));
            drafts.push(d);
            q_rows.push(qd);
            prev = d;
        }
        let mut row_logits = Vec::with_capacity(k + 1);
        let mut p_rows = Vec::with_capacity(k + 1);
        for &state in &states {
            let l = logits(self, state);
            p_rows.push(sampling::processed_probs(self.spec.logits(&l), &params));
            row_logits.push(l);
        }
        let accepted = sampling::chain_accept(&p_rows, &q_rows, &drafts, &params, last as u64 + 1);
        row_logits.truncate(accepted.len());
        Ok((accepted, row_logits))
    }
}

/// The drafted token for the position after `q` from its distribution.
pub(super) fn draw_draft(qd: &[f64], params: &SamplingParams, q: u32) -> u32 {
    if params.is_greedy() {
        qd.iter().position(|&v| v == 1.0).expect("one-hot") as u32
    } else {
        sampling::sample_from(
            qd,
            sampling::uniform(params.seed(), q as u64 + 1, Stream::Draft),
        )
    }
}
