//! Speculative decoding on the device: MTP drafting, verification of the
//! drafts by the target, and chain acceptance, inside one step with no host
//! round trip (`eidola-engine`'s executor contract; the CPU reference
//! executor, `eidola-engine-cpu`, is the oracle this reproduces).
//!
//! # The drafter rows
//!
//! MTP depth `d`'s row at position `s` consumes the token at `s` and the
//! **target's** hidden state at the anchor `a = s - 1 - d`, uses RoPE
//! position `a`, writes its KV at `s` in the drafter group, and predicts the
//! token at `s + 1`; rows exist for `s >= d + 1`. The draft for `p + 1 + i`
//! (after the last host position `p`) is depth `i`'s prediction at `p + i`,
//! anchored at `p - 1` like every depth's. No depth reads another's output:
//! that is how the model vendor serves its MTP layers (layer `d` at anchor
//! `x` combines `h_x` with `t_{x+d+1}`; `eidola-engine-cpu/AGENTS.md` → MTP
//! row layout). Level `l` at `x` names the target's state at `x - l` (zeros
//! before position 0), so the slot state and the taps keep levels `0 .. D`
//! at a position. Which hidden state that is (the final-norm output or the
//! residual stream before it) is [`MtpHidden`].
//!
//! # One step
//!
//! A row that drafts has exactly one host token (a decode row: the serving
//! core drafts only those), so everything its drafter needs comes from
//! earlier steps (every draft row is anchored at or before `p - 1`), and the
//! target runs once over the host tokens and the drafts together. The step,
//! in launch order:
//!
//! 1. **Load**: every row's levels at `c - 1` (`c` its first host position)
//!    are copied into the level buffer, from the slot's state when it was
//!    left at `c - 1`, or from the boundary tap of the drafter block ending
//!    at `c - 1` (a fresh slot after a prefix hit or a resume, which the host
//!    makes only at block boundaries).
//! 2. **Draft**, depth by depth for the drafting rows: depth `i` runs at `p`
//!    and, while `i < k`, at the chain positions `p + 1 ..= p + i` (consuming
//!    drafts `1 ..= i`); its logits at `p + i` give draft `i + 1`, drawn on
//!    the device (stream `Draft` at position `p + 1 + i`, the argmax for
//!    greedy rows), its distribution kept for acceptance.
//! 3. **Verify**: the target over every row's host tokens and drafts, KV
//!    written for all of them. One copy launch then keeps its states for the
//!    fill phase (the target buffer, since every MTP forward reuses the
//!    forward scratch) and stores every level at each position the row may
//!    continue from (`p ..= p + k`; the host knows after the step which) into
//!    the slot's state, and at each block-final position into the block's
//!    tap. A tap written for a rejected draft's position is rewritten when
//!    that position is computed again, before its block can be sealed.
//! 4. **Accept**: target distributions at `p ..= p + k` and chain acceptance
//!    (the bonus or replacement token included); rows without drafts sample
//!    their one token.
//! 5. **Fill**: depth by depth, the drafter rows not computed yet: every
//!    host position of rows without drafts, and `p + 1 ..= p + k` of drafting
//!    rows (less the chain rows step 2 computed), each anchored at a state
//!    the target computed this step or at a loaded level. Rows past the
//!    accepted drafts write KV at positions the host reserved for drafts and
//!    are rewritten when those positions are computed again.
//!
//! # Device data
//!
//! Every per-step value is data in one table of 32-bit words (token ids,
//! positions, KV targets, attention work lists, logit rows, sampler rows,
//! acceptance rows, the copy lists), uploaded once, and the drafts themselves
//! are written into it by the sampler and copied from it into the forwards'
//! token arrays, so they never leave the device. Copies go through one
//! bounded kernel (`eidola_copy_rows`) over a fixed buffer table (`buf`);
//! draft ids are bounded where they are made and where acceptance reads them
//! (`STATUS_BAD_TOKEN`). A uniform drafted decode step (every row one host
//! token, sampled, the full draft width, past the first `D` positions) has a
//! launch shape that is a function of its row count alone, so it can be
//! padded to a rung and replayed as a captured graph (`DraftGraphs`).

use eidola_engine::sampling::{SamplingParams, Stream};
use eidola_engine::spec::AttentionKind;

use crate::attention::AttnRequest;
use crate::sampler::SampleRow;

/// Which of the target's hidden states every MTP depth conditions on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MtpHidden {
    /// The state after the target's final `norm`: what vLLM and SGLang feed.
    #[default]
    Normed,
    /// The state before it: what llama.cpp feeds.
    PreNorm,
}

/// The copy kernel's buffer table for a drafted step: every row copy names
/// one of these.
pub(crate) mod buf {
    /// The last forward's final-norm outputs (rows of `hidden` f32).
    pub const NORMED: u32 = 0;
    /// The last forward's residual stream.
    pub const PRENORM: u32 = 1;
    /// The MTP forward's input: the target state each row is anchored at.
    pub const INPUT: u32 = 2;
    /// The step's level rows ([`super::Levels`]).
    pub const LEVELS: u32 = 3;
    /// Per-slot drafter state ([`super::state_row`]).
    pub const STATE: u32 = 4;
    /// The drafter group's boundary taps ([`super::tap_row`]).
    pub const TAPS: u32 = 5;
    /// The step table, one word per row (token copies).
    pub const TABLE: u32 = 6;
    /// The target's states over the step's tokens ([`super::DraftBuffers`]).
    pub const TARGET: u32 = 7;
}

/// Row `level` of `entry` of `slot`'s state: entry `j` holds the levels at
/// the position `j` past the last host position of the slot's last step.
pub(crate) fn state_row(depths: u32, slot: u32, entry: u32, level: u32) -> u32 {
    (slot * (depths + 1) + entry) * depths + level
}

/// Row `level` of block `block`'s boundary tap.
pub(crate) fn tap_row(depths: u32, block: u32, level: u32) -> u32 {
    block * depths + level
}

/// The level buffer's rows, for at most `rows` rows a step and `depths`
/// draft depths: a zero row, each row's loaded levels, and write-only rows
/// padding and graph steps store into instead of a tap or a state row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Levels {
    pub rows: u32,
    pub depths: u32,
}

impl Levels {
    pub const ZERO: u32 = 0;

    /// Level `level` of row `r` at its first host position less one.
    pub fn load(&self, r: u32, level: u32) -> u32 {
        1 + r * self.depths + level
    }

    /// The first write-only row.
    pub fn dummy(&self) -> u32 {
        1 + self.rows * self.depths
    }

    /// Write-only rows the store launch can need: two per level of each of
    /// at most `depths + 1` positions of every row.
    pub fn dummies(&self) -> u32 {
        2 * self.rows * (self.depths + 1) * self.depths
    }

    /// Every row.
    pub fn total(&self) -> u32 {
        self.dummy() + self.dummies()
    }
}

/// Where a row's levels at `c - 1` come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Load {
    /// The row starts at position 0: nothing precedes it.
    None,
    /// The slot's own state, entry `entry`.
    State { entry: u32 },
    /// The boundary tap of the drafter block ending at `c - 1`.
    Tap { block: u32 },
    /// Zeros (padding rows).
    Zero,
}

/// One row of a drafted step, as the planner sees it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Row {
    /// The state slot, or `None` for a padding row.
    pub slot: Option<u32>,
    /// First host position.
    pub c: u32,
    /// Host tokens (positions `c ..= p`).
    pub tokens: Vec<u32>,
    /// Drafts this step (`0` for a row that does not sample).
    pub k: u32,
    pub sample: bool,
    pub sampling: SamplingParams,
    pub load: Load,
}

impl Row {
    /// The last host position.
    pub fn p(&self) -> u32 {
        self.c + u32::try_from(self.tokens.len()).expect("host tokens fit u32") - 1
    }
}

/// How the planner addresses KV: every write and read of a real row goes
/// through the block tables (panicking on a read of an unmapped block and on
/// a write into a shared one, the seam's contract checks); a padding row's
/// go to each group's pad block.
pub(crate) trait KvMap {
    /// Target group `group`'s block and offset for a write at `pos`.
    fn target(&self, slot: Option<u32>, group: usize, pos: u32) -> (u32, u32);
    /// Target group `group`'s blocks for `pages` (logical block indices), for
    /// reading.
    fn pages(
        &self,
        slot: Option<u32>,
        group: usize,
        pages: std::ops::RangeInclusive<u32>,
    ) -> Vec<u32>;
    /// The drafter plane row of `pos`, for a write (`write`) or a read.
    fn drafter_row(&self, slot: Option<u32>, pos: u32, write: bool) -> u32;
    /// The drafter block holding `pos`, for writing its tap.
    fn tap_block(&self, slot: Option<u32>, pos: u32) -> u32;
}

/// One target KV group as the planner sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TargetGroup {
    pub attention: AttentionKind,
    /// Query heads per KV head.
    pub group_size: u32,
}

/// What the planner needs besides the rows.
#[derive(Clone, Debug)]
pub(crate) struct PlanCtx {
    pub depths: u32,
    pub hidden: MtpHidden,
    pub block_size: u32,
    pub targets: Vec<TargetGroup>,
    /// The drafter group's window (`Sliding`) and GQA group size.
    pub drafter_window: u32,
    pub drafter_group_size: u32,
    /// Rows the step's regions are sized for (lanes, level rows, sampler
    /// rows): the largest step's row count.
    pub cap_rows: u32,
    /// Blocks of the longest sequence (a global request's page bound).
    pub max_blocks: u32,
    /// A graph step: every row position `p ..= p + k` stores a tap or a
    /// write-only row, so the copy counts do not depend on block alignment.
    pub graph: bool,
}

/// Where a copy reads or writes: a row of a buffer, or a word of one of the
/// step table's arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Loc {
    Buf(u32, u32),
    /// Lane `lane` (draft `lane + 1`), drafting row `rd`.
    Lane(u32, u32),
    /// Word `index` of the token array of op `op`.
    Tokens(usize, u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyItem {
    pub src: Loc,
    pub dst: Loc,
}

/// How wide a copy's rows are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Width {
    /// Rows of `hidden` f32.
    Hidden,
    /// One token id.
    Token,
}

/// Where a sampler launch writes its tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TokenOut {
    /// Draft lane `i`, one word per drafting row.
    Lane(u32),
    /// The read-back's plain tokens.
    Plain,
    /// Nowhere (distributions only).
    None,
}

/// One planned launch (or launch group: a forward).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Op {
    Copy {
        width: Width,
        items: Vec<CopyItem>,
    },
    /// MTP depth `depth` over rows (positions are RoPE positions, the
    /// anchors `s - 1 - depth`).
    Mtp {
        depth: u32,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        kv_rows: Vec<u32>,
        requests: Vec<AttnRequest>,
        /// A bound on any one request's pages (graph tables reserve it).
        page_bound: u32,
        logit_rows: Vec<u32>,
    },
    /// The target over host tokens and drafts.
    Target {
        tokens: Vec<u32>,
        positions: Vec<u32>,
        kv_targets: Vec<Vec<(u32, u32)>>,
        requests: Vec<Vec<AttnRequest>>,
        page_bounds: Vec<u32>,
        logit_rows: Vec<u32>,
        final_norm: bool,
    },
    Sample {
        rows: Vec<SampleRow>,
        draw: Option<Stream>,
        /// The first distribution row it writes.
        probs_row: u32,
        tokens: TokenOut,
    },
    Accept {
        rows: Vec<SampleRow>,
        target_row: Vec<u32>,
        draft_row: Vec<u32>,
        num_drafts: Vec<u32>,
    },
}

/// Distribution rows of the sampler's scratch, for `cap_rows` rows and
/// `depths` depths: draft distributions (lane-major, as the drafts), target
/// distributions (and plain rows'), acceptance's residuals.
pub(crate) fn probs_rows(cap_rows: u32, depths: u32) -> u32 {
    (2 * depths + 2) * cap_rows
}

/// A planned step: its launches in order, and where its outputs land.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Plan {
    pub ops: Vec<Op>,
    /// Per row: its drafting index (lane column, acceptance row) when it
    /// drafts, or its plain-token index when it samples without drafts.
    pub out: Vec<RowOut>,
    /// Per row: the target logits row of its last host position (sampled
    /// rows), the drafts' rows following it.
    pub logits: Vec<Option<u32>>,
    pub drafting_rows: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowOut {
    Drafted(u32),
    Plain(u32),
    None,
}

/// The planner: every launch of a drafted step over `rows`, in order (see
/// the module docs). Panics on a contract violation (through `kv`), before
/// anything is launched.
pub(crate) fn plan(rows: &[Row], ctx: &PlanCtx, kv: &dyn KvMap) -> Plan {
    let d_count = ctx.depths;
    assert!(d_count > 0, "a drafted step needs draft depths");
    let bs = ctx.block_size;
    let r_count = u32::try_from(rows.len()).expect("rows fit u32");
    assert!(
        r_count <= ctx.cap_rows,
        "{r_count} rows past {}",
        ctx.cap_rows
    );
    let lv = Levels {
        rows: ctx.cap_rows,
        depths: d_count,
    };
    let level_buf = match ctx.hidden {
        MtpHidden::Normed => buf::NORMED,
        MtpHidden::PreNorm => buf::PRENORM,
    };
    // Drafting index of each row.
    let mut rd_of = vec![None; rows.len()];
    let mut drafting = 0u32;
    for (r, row) in rows.iter().enumerate() {
        assert!(!row.tokens.is_empty(), "a row needs a host token");
        if row.k > 0 {
            assert!(row.sample, "drafts on a row that does not sample");
            assert!(
                row.tokens.len() == 1,
                "drafts on a row of {} host tokens: this executor drafts only decode rows",
                row.tokens.len()
            );
            assert!(row.p() > 0, "{} drafts requested at position 0", row.k);
            assert!(
                row.k <= d_count,
                "{} drafts; the drafter has {d_count} depths",
                row.k
            );
            rd_of[r] = Some(drafting);
            drafting += 1;
        }
        assert_eq!(
            row.load == Load::None,
            row.c == 0,
            "levels are loaded exactly for rows past position 0"
        );
    }
    let mut ops: Vec<Op> = Vec::new();
    let dummy = |dummies: &mut u32| {
        let at = lv.dummy() + *dummies;
        *dummies += 1;
        assert!(*dummies <= lv.dummies(), "write-only rows exhausted");
        Loc::Buf(buf::LEVELS, at)
    };
    let state_at = |row: &Row, entry: u32, level: u32| -> Option<Loc> {
        row.slot
            .map(|slot| Loc::Buf(buf::STATE, state_row(d_count, slot, entry, level)))
    };
    // A tap store for `level` at `x` of `row`, when `x` ends a block; in a
    // graph step every position `p ..= p + k` stores one (or a write-only
    // row), as does every position of a padding row.
    let tap_at = |row: &Row, x: u32, level: u32| -> Option<Loc> {
        match row.slot {
            Some(_) if (x + 1).is_multiple_of(bs) => Some(Loc::Buf(
                buf::TAPS,
                tap_row(d_count, kv.tap_block(row.slot, x), level),
            )),
            _ => None,
        }
    };

    // 1. Load.
    let mut items = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        let r32 = u32::try_from(r).expect("rows fit u32");
        for l in 0..d_count {
            let src = match row.load {
                Load::None => continue,
                Load::State { entry } => Loc::Buf(
                    buf::STATE,
                    state_row(d_count, row.slot.expect("a real row"), entry, l),
                ),
                Load::Tap { block } => Loc::Buf(buf::TAPS, tap_row(d_count, block, l)),
                Load::Zero => Loc::Buf(buf::LEVELS, Levels::ZERO),
            };
            items.push(CopyItem {
                src,
                dst: Loc::Buf(buf::LEVELS, lv.load(r32, l)),
            });
        }
    }
    push_copy(&mut ops, Width::Hidden, items);

    // The drafter's KV request for one row's run of queries at `first ..=
    // last` of depth `depth`: one-position pages from the first position its
    // first query sees (and depth `depth` has, `depth + 1` on) to `last`.
    let mtp_request = |row: &Row, depth: u32, q_start: u32, first: u32, last: u32| {
        let window = AttentionKind::Sliding {
            window: ctx.drafter_window,
        };
        let lo = window.first_visible(first).max(depth + 1);
        AttnRequest {
            q_start,
            qo_len: last - first + 1,
            pages: (lo..=last)
                .map(|x| kv.drafter_row(row.slot, x, false))
                .collect(),
            kv_len: last - lo + 1,
        }
    };
    let mtp_page_bound = ctx.drafter_window - 1 + d_count + 1;

    // 2. Draft: depth i at p and the chain positions p + 1 ..= p + i (i < k),
    // every row anchored at or before c - 1 = p - 1, so its target state is
    // a loaded level.
    for i in 0..d_count {
        let mut items: Vec<(usize, u32, u32)> = Vec::new();
        let mut hp = Vec::new();
        let mut tok_copies = Vec::new();
        let (mut tokens, mut positions, mut kv_rows, mut requests) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut logit_rows = Vec::new();
        let mut samples = Vec::new();
        let op_index = ops.len() + 2;
        for (r, row) in rows.iter().enumerate() {
            let Some(rd) = rd_of[r] else { continue };
            let p = row.p();
            let jmax = if i < row.k { i } else { 0 };
            let start = u32::try_from(items.len()).expect("items fit u32");
            for j in 0..=jmax {
                let s = p + j;
                if s < i + 1 {
                    continue;
                }
                let n = u32::try_from(items.len()).expect("items fit u32");
                items.push((r, s, n));
                let r32 = u32::try_from(r).expect("rows fit u32");
                // The anchor s - 1 - i is p - 1 - (i - j): level i - j at p - 1.
                hp.push(CopyItem {
                    src: Loc::Buf(buf::LEVELS, lv.load(r32, i - j)),
                    dst: Loc::Buf(buf::INPUT, n),
                });
                if j == 0 {
                    tokens.push(row.tokens[0]);
                } else {
                    tokens.push(0);
                    tok_copies.push(CopyItem {
                        src: Loc::Lane(j - 1, rd),
                        dst: Loc::Tokens(op_index, n),
                    });
                }
                positions.push(s - 1 - i);
                kv_rows.push(kv.drafter_row(row.slot, s, true));
            }
            let end = u32::try_from(items.len()).expect("items fit u32");
            if end > start {
                let first = items[start as usize].1;
                requests.push(mtp_request(row, i, start, first, p + jmax));
            }
            // The draft sampled from this depth: the row's last item's
            // logits (its prediction at p + i while i < k).
            let wanted = i < row.k && end > start;
            if wanted {
                logit_rows.push(end - 1);
            }
            let logit_row = if wanted {
                u32::try_from(logit_rows.len() - 1).expect("rows fit u32")
            } else {
                0
            };
            samples.push(if wanted && row.slot.is_some() {
                SampleRow::new(&row.sampling, p + 1 + i, logit_row)
            } else {
                // A row this depth does not draft for (or padding): greedy
                // over logits row 0, its token and distribution unused.
                SampleRow {
                    logit_row: 0,
                    ..SampleRow::default()
                }
            });
        }
        if items.is_empty() {
            continue;
        }
        push_copy(&mut ops, Width::Hidden, hp);
        // Keep op indices stable: the token copy is always an op (possibly
        // empty) right before the forward.
        ops.push(Op::Copy {
            width: Width::Token,
            items: tok_copies,
        });
        debug_assert_eq!(ops.len(), op_index);
        ops.push(Op::Mtp {
            depth: i,
            tokens,
            positions,
            kv_rows,
            requests,
            page_bound: mtp_page_bound,
            logit_rows: logit_rows.clone(),
        });
        if !logit_rows.is_empty() {
            ops.push(Op::Sample {
                rows: samples,
                draw: Some(Stream::Draft),
                probs_row: i * ctx.cap_rows,
                tokens: TokenOut::Lane(i),
            });
        }
    }

    // 3. Verify: the target over host tokens and drafts.
    let mut target_items: Vec<(usize, u32, u32)> = Vec::new();
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut kv_targets = vec![Vec::new(); ctx.targets.len()];
    let mut requests = vec![Vec::new(); ctx.targets.len()];
    let mut tok_copies = Vec::new();
    let mut logit_rows = Vec::new();
    let mut logits = vec![None; rows.len()];
    let op_index = ops.len() + 1;
    for (r, row) in rows.iter().enumerate() {
        let (c, p) = (row.c, row.p());
        let last = p + row.k;
        let q_start = u32::try_from(target_items.len()).expect("items fit u32");
        for x in c..=last {
            let n = u32::try_from(target_items.len()).expect("items fit u32");
            target_items.push((r, x, n));
            if x <= p {
                tokens.push(row.tokens[(x - c) as usize]);
            } else {
                tokens.push(0);
                tok_copies.push(CopyItem {
                    src: Loc::Lane(x - p - 1, rd_of[r].expect("drafts on a drafting row")),
                    dst: Loc::Tokens(op_index, n),
                });
            }
            positions.push(x);
            for (g, kvt) in kv_targets.iter_mut().enumerate() {
                kvt.push(kv.target(row.slot, g, x));
            }
            if row.sample && x >= p {
                if x == p {
                    logits[r] = Some(u32::try_from(logit_rows.len()).expect("rows fit u32"));
                }
                logit_rows.push(n);
            }
        }
        for (g, group) in ctx.targets.iter().enumerate() {
            let first_page = group.attention.first_visible(c) / bs;
            requests[g].push(AttnRequest {
                q_start,
                qo_len: last - c + 1,
                pages: kv.pages(row.slot, g, first_page..=last / bs),
                kv_len: last + 1 - first_page * bs,
            });
        }
    }
    // A row of one host token and at most `D` drafts sees at most `W - 1 +
    // D + 1` positions of a window `W`; every block for global attention.
    let page_bounds = ctx
        .targets
        .iter()
        .map(|g| match g.attention {
            AttentionKind::Full => ctx.max_blocks,
            AttentionKind::Sliding { window } => {
                ((window - 1 + d_count).div_ceil(bs) + 1).min(ctx.max_blocks)
            }
        })
        .collect();
    ops.push(Op::Copy {
        width: Width::Token,
        items: tok_copies,
    });
    debug_assert_eq!(ops.len(), op_index);
    ops.push(Op::Target {
        tokens,
        positions,
        kv_targets,
        requests,
        page_bounds,
        logit_rows: logit_rows.clone(),
        final_norm: ctx.hidden == MtpHidden::Normed,
    });
    // The target's states, in one launch: every computed position's state
    // into the target buffer the fill phase reads (the MTP forwards reuse
    // the forward scratch), and every level at each position the row may
    // continue from (`p ..= p + k`) or that ends a block into the slot's
    // state and the block's tap. Level `l` at `x` is the target's state at
    // `x - l`: this forward's when `x - l >= c`, a loaded level below it,
    // zeros before position 0.
    let mut out = Vec::new();
    let mut dummies = 0u32;
    for &(r, x, n) in &target_items {
        out.push(CopyItem {
            src: Loc::Buf(level_buf, n),
            dst: Loc::Buf(buf::TARGET, n),
        });
        let row = &rows[r];
        let (c, p) = (row.c, row.p());
        let in_state = x >= p && x <= p + row.k;
        let tap = (x + 1).is_multiple_of(bs);
        if !in_state && !(tap && row.slot.is_some()) {
            continue;
        }
        let r32 = u32::try_from(r).expect("rows fit u32");
        for l in 0..d_count {
            let src = match x.checked_sub(l) {
                None => Loc::Buf(buf::LEVELS, Levels::ZERO),
                Some(a) if a >= c => Loc::Buf(level_buf, n - l),
                Some(a) => Loc::Buf(buf::LEVELS, lv.load(r32, c - 1 - a)),
            };
            if in_state {
                out.push(CopyItem {
                    src,
                    dst: state_at(row, x - p, l).unwrap_or_else(|| dummy(&mut dummies)),
                });
            }
            match tap_at(row, x, l) {
                Some(dst) => out.push(CopyItem { src, dst }),
                None if in_state && (ctx.graph || row.slot.is_none()) => out.push(CopyItem {
                    src,
                    dst: dummy(&mut dummies),
                }),
                None => {}
            }
        }
    }
    push_copy(&mut ops, Width::Hidden, out);

    // 4. Accept, and plain samples.
    let mut out = vec![RowOut::None; rows.len()];
    let pt_base = d_count * ctx.cap_rows;
    let mut pt_rows = Vec::new();
    let (mut acc_rows, mut target_row, mut draft_row, mut num_drafts) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (r, row) in rows.iter().enumerate() {
        let Some(rd) = rd_of[r] else { continue };
        let first = logits[r].expect("a drafting row samples");
        target_row.push(pt_base + u32::try_from(pt_rows.len()).expect("rows fit u32"));
        for j in 0..=row.k {
            pt_rows.push(if row.slot.is_some() {
                SampleRow::new(&row.sampling, row.p() + 1 + j, first + j)
            } else {
                SampleRow {
                    logit_row: 0,
                    ..SampleRow::default()
                }
            });
        }
        acc_rows.push(if row.slot.is_some() {
            SampleRow::new(&row.sampling, row.p() + 1, 0)
        } else {
            SampleRow::default()
        });
        draft_row.push(rd);
        num_drafts.push(row.k);
        out[r] = RowOut::Drafted(rd);
    }
    let drafted_pt = u32::try_from(pt_rows.len()).expect("rows fit u32");
    if !pt_rows.is_empty() {
        ops.push(Op::Sample {
            rows: pt_rows,
            draw: None,
            probs_row: pt_base,
            tokens: TokenOut::None,
        });
        ops.push(Op::Accept {
            rows: acc_rows,
            target_row,
            draft_row,
            num_drafts,
        });
    }
    let mut plain = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        if row.sample && row.k == 0 {
            out[r] = RowOut::Plain(u32::try_from(plain.len()).expect("rows fit u32"));
            plain.push(SampleRow::new(
                &row.sampling,
                row.p() + 1,
                logits[r].expect("a sampled row has logits"),
            ));
        }
    }
    if !plain.is_empty() {
        ops.push(Op::Sample {
            rows: plain,
            draw: Some(Stream::Sample),
            probs_row: pt_base + drafted_pt,
            tokens: TokenOut::Plain,
        });
    }

    // 5. Fill: the drafter rows not computed yet, depth by depth, each
    // anchored at a position this step's target computed or a loaded level.
    for d in 0..d_count {
        let mut items: Vec<(usize, u32, u32)> = Vec::new();
        let mut hp = Vec::new();
        let mut tok_copies = Vec::new();
        let (mut tokens, mut positions, mut kv_rows, mut requests) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let op_index = ops.len() + 2;
        let target_n = |r: usize, x: u32| -> u32 {
            target_items
                .iter()
                .find(|&&(rr, xx, _)| rr == r && xx == x)
                .map(|&(_, _, n)| n)
                .expect("the target computed this position")
        };
        for (r, row) in rows.iter().enumerate() {
            let r32 = u32::try_from(r).expect("rows fit u32");
            let p = row.p();
            let done = |depth: u32| {
                if depth >= 1 && depth < row.k {
                    depth
                } else {
                    0
                }
            };
            let (lo, hi) = match rd_of[r] {
                Some(_) => ((p + 1 + done(d)).max(d + 1), p + row.k),
                None => (row.c.max(d + 1), p),
            };
            if lo > hi {
                continue;
            }
            let start = u32::try_from(items.len()).expect("items fit u32");
            for s in lo..=hi {
                let n = u32::try_from(items.len()).expect("items fit u32");
                items.push((r, s, n));
                let a = s - 1 - d;
                let src = if a >= row.c {
                    Loc::Buf(buf::TARGET, target_n(r, a))
                } else {
                    Loc::Buf(buf::LEVELS, lv.load(r32, row.c - 1 - a))
                };
                hp.push(CopyItem {
                    src,
                    dst: Loc::Buf(buf::INPUT, n),
                });
                if s <= p {
                    tokens.push(row.tokens[(s - row.c) as usize]);
                } else {
                    tokens.push(0);
                    tok_copies.push(CopyItem {
                        src: Loc::Lane(s - p - 1, rd_of[r].expect("a drafting row")),
                        dst: Loc::Tokens(op_index, n),
                    });
                }
                positions.push(a);
                kv_rows.push(kv.drafter_row(row.slot, s, true));
            }
            requests.push(mtp_request(row, d, start, lo, hi));
        }
        if items.is_empty() {
            continue;
        }
        push_copy(&mut ops, Width::Hidden, hp);
        ops.push(Op::Copy {
            width: Width::Token,
            items: tok_copies,
        });
        debug_assert_eq!(ops.len(), op_index);
        ops.push(Op::Mtp {
            depth: d,
            tokens,
            positions,
            kv_rows,
            requests,
            page_bound: mtp_page_bound,
            logit_rows: Vec::new(),
        });
    }
    Plan {
        ops,
        out,
        logits,
        drafting_rows: drafting,
    }
}

/// A copy op, unless it has no items. Hidden-width copies with no items
/// are left out (a graph step's shape still matches, since its counts do
/// not vary); token copies are kept, so a forward's token array is always
/// the op right after its token copy.
fn push_copy(ops: &mut Vec<Op>, width: Width, items: Vec<CopyItem>) {
    if !items.is_empty() {
        ops.push(Op::Copy { width, items });
    }
}

/// One array of the step table, as the host builds it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum HostArray {
    Words(Vec<u32>),
    /// A copy list's buffer ids or rows, resolved against the layout when
    /// packed (lanes and token arrays live in the table itself).
    Locs {
        locs: Vec<Loc>,
        rows: bool,
    },
}

impl HostArray {
    fn len(&self) -> usize {
        match self {
            HostArray::Words(w) => w.len(),
            HostArray::Locs { locs, .. } => locs.len(),
        }
    }
}

/// A table array: its contents, the words a graph table reserves for it
/// (page lists vary from step to step; everything else is a function of the
/// shape), and its alignment in words.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Array {
    pub data: HostArray,
    pub page_cap: Option<usize>,
    pub align: usize,
}

/// One attention work list, by array index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanSpec {
    pub tile: u32,
    pub page_size: u32,
    pub group_size: u32,
    pub work_items: u32,
    pub num_requests: u32,
    /// q_indptr, indices, indptr, last_page_len, request, query-tile and
    /// KV-tile indices.
    pub arrays: [usize; 7],
}

/// One launch, its per-step values named by array index (see [`Launch`]
/// for the same with table offsets).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Spec {
    Copy {
        hidden: bool,
        items: u32,
        /// Source buffers, source rows, destination buffers, destination
        /// rows.
        arrays: [usize; 4],
    },
    Target {
        tokens: u32,
        token_ids: usize,
        positions: usize,
        kv_block: Vec<usize>,
        kv_slot: Vec<usize>,
        plans: Vec<PlanSpec>,
        logit_rows: usize,
        num_logit_rows: u32,
        final_norm: bool,
    },
    Mtp {
        depth: u32,
        tokens: u32,
        token_ids: usize,
        positions: usize,
        kv_rows: usize,
        plan: PlanSpec,
        logit_rows: usize,
        num_logit_rows: u32,
    },
    Sample {
        rows: usize,
        num_rows: u32,
        draw: Option<Stream>,
        probs_row: u32,
        tokens: TokenOut,
    },
    Accept {
        rows: usize,
        target_row: usize,
        draft_row: usize,
        num_drafts: usize,
        num_rows: u32,
    },
}

/// A plan as table arrays and launches over them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Built {
    pub arrays: Vec<Array>,
    pub specs: Vec<Spec>,
    /// Per op index: the array holding its token ids (forwards only).
    pub token_arrays: Vec<Option<usize>>,
}

/// The arrays and launches of `plan` (attention work lists made, and
/// checked, by [`HostPlan::new`]).
pub(crate) fn build(plan: &Plan, ctx: &PlanCtx) -> crate::Result<Built> {
    use crate::attention::HostPlan;
    let mut arrays: Vec<Array> = Vec::new();
    let push = |arrays: &mut Vec<Array>, data: HostArray, page_cap: Option<usize>, align: usize| {
        arrays.push(Array {
            data,
            page_cap,
            align,
        });
        arrays.len() - 1
    };
    let words = |v: &[u32]| HostArray::Words(v.to_vec());
    let n32 = |n: usize| u32::try_from(n).map_err(|_| crate::CudaError::new("plan past u32"));
    let attn = |arrays: &mut Vec<Array>,
                requests: &[AttnRequest],
                group_size: u32,
                page_size: u32,
                page_bound: u32|
     -> crate::Result<PlanSpec> {
        let h = HostPlan::new(requests, group_size, page_size)?;
        let w = |v: &[i32]| {
            HostArray::Words(
                v.iter()
                    .map(|&x| u32::try_from(x).expect("HostPlan indices are non-negative"))
                    .collect(),
            )
        };
        let cap = requests.len() * page_bound as usize;
        let a = [
            push(arrays, w(&h.q_indptr), None, 1),
            push(arrays, w(&h.indices), Some(cap), 1),
            push(arrays, w(&h.indptr), None, 1),
            push(arrays, w(&h.last_page_len), None, 1),
            push(arrays, w(&h.request_indices), None, 1),
            push(arrays, w(&h.qo_tile_indices), None, 1),
            push(arrays, w(&h.kv_tile_indices), None, 1),
        ];
        Ok(PlanSpec {
            tile: h.tile(),
            page_size,
            group_size,
            work_items: n32(h.work_items())?,
            num_requests: n32(h.num_requests())?,
            arrays: a,
        })
    };
    let sample_words = |rows: &[SampleRow]| {
        HostArray::Words(
            rows.iter()
                .flat_map(crate::graph::sample_row_words)
                .collect(),
        )
    };
    let mut specs = Vec::with_capacity(plan.ops.len());
    let mut token_arrays = vec![None; plan.ops.len()];
    for (o, op) in plan.ops.iter().enumerate() {
        specs.push(match op {
            Op::Copy { width, items } => {
                let col = |f: &dyn Fn(&CopyItem) -> Loc, rows: bool| HostArray::Locs {
                    locs: items.iter().map(f).collect(),
                    rows,
                };
                Spec::Copy {
                    hidden: *width == Width::Hidden,
                    items: n32(items.len())?,
                    arrays: [
                        push(&mut arrays, col(&|i| i.src, false), None, 1),
                        push(&mut arrays, col(&|i| i.src, true), None, 1),
                        push(&mut arrays, col(&|i| i.dst, false), None, 1),
                        push(&mut arrays, col(&|i| i.dst, true), None, 1),
                    ],
                }
            }
            Op::Mtp {
                depth,
                tokens,
                positions,
                kv_rows,
                requests,
                page_bound,
                logit_rows,
            } => {
                let token_ids = push(&mut arrays, words(tokens), None, 1);
                token_arrays[o] = Some(token_ids);
                Spec::Mtp {
                    depth: *depth,
                    tokens: n32(tokens.len())?,
                    token_ids,
                    positions: push(&mut arrays, words(positions), None, 1),
                    kv_rows: push(&mut arrays, words(kv_rows), None, 1),
                    plan: attn(
                        &mut arrays,
                        requests,
                        ctx.drafter_group_size,
                        1,
                        *page_bound,
                    )?,
                    logit_rows: push(&mut arrays, words(logit_rows), None, 1),
                    num_logit_rows: n32(logit_rows.len())?,
                }
            }
            Op::Target {
                tokens,
                positions,
                kv_targets,
                requests,
                page_bounds,
                logit_rows,
                final_norm,
            } => {
                let token_ids = push(&mut arrays, words(tokens), None, 1);
                token_arrays[o] = Some(token_ids);
                let positions = push(&mut arrays, words(positions), None, 1);
                let mut kv_block = Vec::new();
                let mut kv_slot = Vec::new();
                let mut plans = Vec::new();
                for (g, group) in ctx.targets.iter().enumerate() {
                    let b: Vec<u32> = kv_targets[g].iter().map(|t| t.0).collect();
                    let s: Vec<u32> = kv_targets[g].iter().map(|t| t.1).collect();
                    kv_block.push(push(&mut arrays, words(&b), None, 1));
                    kv_slot.push(push(&mut arrays, words(&s), None, 1));
                    plans.push(attn(
                        &mut arrays,
                        &requests[g],
                        group.group_size,
                        ctx.block_size,
                        page_bounds[g],
                    )?);
                }
                Spec::Target {
                    tokens: n32(tokens.len())?,
                    token_ids,
                    positions,
                    kv_block,
                    kv_slot,
                    plans,
                    logit_rows: push(&mut arrays, words(logit_rows), None, 1),
                    num_logit_rows: n32(logit_rows.len())?,
                    final_norm: *final_norm,
                }
            }
            Op::Sample {
                rows,
                draw,
                probs_row,
                tokens,
            } => Spec::Sample {
                rows: push(&mut arrays, sample_words(rows), None, 2),
                num_rows: n32(rows.len())?,
                draw: *draw,
                probs_row: *probs_row,
                tokens: *tokens,
            },
            Op::Accept {
                rows,
                target_row,
                draft_row,
                num_drafts,
            } => Spec::Accept {
                rows: push(&mut arrays, sample_words(rows), None, 2),
                target_row: push(&mut arrays, words(target_row), None, 1),
                draft_row: push(&mut arrays, words(draft_row), None, 1),
                num_drafts: push(&mut arrays, words(num_drafts), None, 1),
                num_rows: n32(rows.len())?,
            },
        });
    }
    Ok(Built {
        arrays,
        specs,
        token_arrays,
    })
}

/// Where everything sits in the step table, in 32-bit words: the draft
/// lanes (`depths × cap_rows`, written by the sampler), one zero word
/// (FlashInfer's KV chunk size, read only when splitting KV), then every
/// array in order, each at its alignment, with room for its reservation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub lanes: usize,
    pub kv_chunk: usize,
    /// Per array: `(offset, words reserved)`.
    pub arrays: Vec<(usize, usize)>,
    pub words: usize,
}

impl Layout {
    /// A layout for `built`: each array exactly its length (`capped`
    /// false, an eager step's table) or with its page reservation (a graph
    /// rung's table, which later steps of the same shape reuse).
    pub fn new(built: &Built, ctx: &PlanCtx, capped: bool) -> crate::Result<Layout> {
        let overflow = || crate::CudaError::new("drafted step table overflows");
        let mut at = 0usize;
        let mut place = |len: usize, align: usize| -> crate::Result<usize> {
            let off = at.next_multiple_of(align);
            at = off.checked_add(len).ok_or_else(overflow)?;
            Ok(off)
        };
        let lanes = place(
            (ctx.depths as usize)
                .checked_mul(ctx.cap_rows as usize)
                .ok_or_else(overflow)?,
            1,
        )?;
        let kv_chunk = place(1, 1)?;
        let mut arrays = Vec::with_capacity(built.arrays.len());
        for a in &built.arrays {
            let cap = match (capped, a.page_cap) {
                (true, Some(c)) => c.max(a.data.len()),
                _ => a.data.len(),
            };
            arrays.push((place(cap, a.align)?, cap));
        }
        Ok(Layout {
            lanes,
            kv_chunk,
            arrays,
            words: at,
        })
    }

    /// The table word of draft `lane + 1` of drafting row `rd`.
    pub fn lane(&self, ctx: &PlanCtx, lane: u32, rd: u32) -> usize {
        self.lanes + lane as usize * ctx.cap_rows as usize + rd as usize
    }
}

/// `built`'s arrays packed into `layout` (each within its reservation; a
/// copy list's table locations resolved to table words). Refuses an array
/// past its reservation, changing nothing.
pub(crate) fn pack(built: &Built, layout: &Layout, ctx: &PlanCtx) -> crate::Result<Vec<u32>> {
    if layout.arrays.len() != built.arrays.len() {
        return Err(crate::CudaError::new(
            "drafted step: the plan's arrays are not the layout's",
        ));
    }
    let mut w = vec![0u32; layout.words];
    for (i, (a, &(at, cap))) in built.arrays.iter().zip(&layout.arrays).enumerate() {
        if a.data.len() > cap {
            return Err(crate::CudaError::new(format!(
                "drafted step: array {i} holds {} words, its reservation {cap}",
                a.data.len()
            )));
        }
        match &a.data {
            HostArray::Words(v) => w[at..at + v.len()].copy_from_slice(v),
            HostArray::Locs { locs, rows } => {
                for (j, loc) in locs.iter().enumerate() {
                    let (b, row) = match *loc {
                        Loc::Buf(b, r) => (b, r as usize),
                        Loc::Lane(lane, rd) => (buf::TABLE, layout.lane(ctx, lane, rd)),
                        Loc::Tokens(op, n) => {
                            let arr = built.token_arrays[op].ok_or_else(|| {
                                crate::CudaError::new(
                                    "drafted step: a copy into an op's tokens it has none of",
                                )
                            })?;
                            (buf::TABLE, layout.arrays[arr].0 + n as usize)
                        }
                    };
                    w[at + j] = if *rows {
                        u32::try_from(row).map_err(|_| {
                            crate::CudaError::new("drafted step table past u32 rows")
                        })?
                    } else {
                        b
                    };
                }
            }
        }
    }
    Ok(w)
}

/// One attention work list as a launch reads it: its shape, and table
/// offsets of its arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanAt {
    pub tile: u32,
    pub page_size: u32,
    pub group_size: u32,
    pub work_items: u32,
    pub num_requests: u32,
    pub arrays: [usize; 7],
}

/// One launch with its per-step values as table offsets (words): what a
/// graph records. Two steps replay one graph exactly when their launch
/// lists are equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Launch {
    Copy {
        hidden: bool,
        items: u32,
        arrays: [usize; 4],
    },
    Target {
        tokens: u32,
        token_ids: usize,
        positions: usize,
        kv_block: Vec<usize>,
        kv_slot: Vec<usize>,
        plans: Vec<PlanAt>,
        logit_rows: usize,
        num_logit_rows: u32,
        final_norm: bool,
    },
    Mtp {
        depth: u32,
        tokens: u32,
        token_ids: usize,
        positions: usize,
        kv_rows: usize,
        plan: PlanAt,
        logit_rows: usize,
        num_logit_rows: u32,
    },
    Sample {
        rows: usize,
        num_rows: u32,
        draw: Option<Stream>,
        probs_row: u32,
        tokens: TokenOut,
    },
    Accept {
        rows: usize,
        target_row: usize,
        draft_row: usize,
        num_drafts: usize,
        num_rows: u32,
    },
}

/// `built`'s launches over `layout`.
pub(crate) fn launches(built: &Built, layout: &Layout) -> Vec<Launch> {
    let at = |i: usize| layout.arrays[i].0;
    let plan = |p: &PlanSpec| PlanAt {
        tile: p.tile,
        page_size: p.page_size,
        group_size: p.group_size,
        work_items: p.work_items,
        num_requests: p.num_requests,
        arrays: p.arrays.map(at),
    };
    built
        .specs
        .iter()
        .map(|s| match s {
            Spec::Copy {
                hidden,
                items,
                arrays,
            } => Launch::Copy {
                hidden: *hidden,
                items: *items,
                arrays: arrays.map(at),
            },
            Spec::Target {
                tokens,
                token_ids,
                positions,
                kv_block,
                kv_slot,
                plans,
                logit_rows,
                num_logit_rows,
                final_norm,
            } => Launch::Target {
                tokens: *tokens,
                token_ids: at(*token_ids),
                positions: at(*positions),
                kv_block: kv_block.iter().map(|&i| at(i)).collect(),
                kv_slot: kv_slot.iter().map(|&i| at(i)).collect(),
                plans: plans.iter().map(plan).collect(),
                logit_rows: at(*logit_rows),
                num_logit_rows: *num_logit_rows,
                final_norm: *final_norm,
            },
            Spec::Mtp {
                depth,
                tokens,
                token_ids,
                positions,
                kv_rows,
                plan: p,
                logit_rows,
                num_logit_rows,
            } => Launch::Mtp {
                depth: *depth,
                tokens: *tokens,
                token_ids: at(*token_ids),
                positions: at(*positions),
                kv_rows: at(*kv_rows),
                plan: plan(p),
                logit_rows: at(*logit_rows),
                num_logit_rows: *num_logit_rows,
            },
            Spec::Sample {
                rows,
                num_rows,
                draw,
                probs_row,
                tokens,
            } => Launch::Sample {
                rows: at(*rows),
                num_rows: *num_rows,
                draw: *draw,
                probs_row: *probs_row,
                tokens: *tokens,
            },
            Spec::Accept {
                rows,
                target_row,
                draft_row,
                num_drafts,
                num_rows,
            } => Launch::Accept {
                rows: at(*rows),
                target_row: at(*target_row),
                draft_row: at(*draft_row),
                num_drafts: at(*num_drafts),
                num_rows: *num_rows,
            },
        })
        .collect()
}

/// Words of a drafted step's read-back: the status word, then per drafting
/// row its `depths + 1` output tokens, then per drafting row its token
/// count, then per plain row its token.
pub(crate) fn readback_words(cap_rows: u32, depths: u32) -> usize {
    1 + cap_rows as usize * (depths as usize + 3)
}

/// The read-back, split.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Readback<'a> {
    pub status: u32,
    /// `cap_rows × (depths + 1)`.
    pub out: &'a [u32],
    pub counts: &'a [u32],
    pub plain: &'a [u32],
}

pub(crate) fn split(words: &[u32], cap_rows: u32, depths: u32) -> Readback<'_> {
    let (cap, stride) = (cap_rows as usize, depths as usize + 1);
    assert_eq!(
        words.len(),
        readback_words(cap_rows, depths),
        "a drafted read-back"
    );
    let (status, rest) = words.split_at(1);
    let (out, rest) = rest.split_at(cap * stride);
    let (counts, plain) = rest.split_at(cap);
    Readback {
        status: status[0],
        out,
        counts,
        plain,
    }
}

/// What drafted steps keep on the device besides the KV store and the
/// model's scratch, allocated once: the level rows, the target's states over
/// a step's tokens, the sampler's distributions, and the read-back.
pub(crate) struct DraftBuffers {
    pub levels: cudarc::driver::CudaSlice<f32>,
    pub target: cudarc::driver::CudaSlice<f32>,
    pub probs: cudarc::driver::CudaSlice<f64>,
    pub readback: cudarc::driver::CudaSlice<u32>,
}

impl DraftBuffers {
    pub fn new(
        gpu: &crate::Gpu,
        ctx: &PlanCtx,
        hidden: usize,
        max_tokens: usize,
        sampleable: u32,
    ) -> crate::Result<Self> {
        let overflow = || crate::CudaError::new("drafted step buffers overflow");
        let lv = Levels {
            rows: ctx.cap_rows,
            depths: ctx.depths,
        };
        let s = gpu.stream();
        Ok(DraftBuffers {
            levels: s.alloc_zeros(
                (lv.total() as usize)
                    .checked_mul(hidden)
                    .ok_or_else(overflow)?,
            )?,
            target: s.alloc_zeros(max_tokens.checked_mul(hidden).ok_or_else(overflow)?)?,
            probs: s.alloc_zeros(
                (probs_rows(ctx.cap_rows, ctx.depths) as usize)
                    .checked_mul(sampleable as usize)
                    .ok_or_else(overflow)?,
            )?,
            readback: s.alloc_zeros(readback_words(ctx.cap_rows, ctx.depths))?,
        })
    }
}

/// What a drafted step's launches run against besides the table: the
/// model, the KV store and the drafting buffers.
pub(crate) struct RunTarget<'a> {
    pub model: &'a mut crate::model::GpuModel,
    pub kv: &'a crate::kv::KvStore,
    pub bufs: &'a mut DraftBuffers,
    pub vocab: u32,
    pub sampleable: u32,
    /// The drafter group's index.
    pub drafter: usize,
}

impl RunTarget<'_> {
    pub fn args<'s>(&'s mut self, ctx: &'s PlanCtx, table: u64, table_words: usize) -> RunArgs<'s> {
        RunArgs {
            model: &mut *self.model,
            kv: self.kv,
            bufs: &mut *self.bufs,
            ctx,
            table,
            table_words,
            vocab: self.vocab,
            sampleable: self.sampleable,
            drafter: self.drafter,
        }
    }
}

/// What a drafted step's launches run against.
pub(crate) struct RunArgs<'a> {
    pub model: &'a mut crate::model::GpuModel,
    pub kv: &'a crate::kv::KvStore,
    pub bufs: &'a mut DraftBuffers,
    pub ctx: &'a PlanCtx,
    /// The step table's device address and words.
    pub table: u64,
    pub table_words: usize,
    pub vocab: u32,
    pub sampleable: u32,
    /// The drafter group's index.
    pub drafter: usize,
}

/// The step's launches, in order, over a table packed for `layout`: the
/// status word cleared, then every launch. Copies nothing from the host,
/// allocates nothing and reads nothing back, so it can be launched directly
/// or recorded.
///
/// # Safety
///
/// The table at `a.table` must hold a step [`pack`]ed for `layout` (whose
/// every array came from [`build`] over a [`plan`] for these buffers, the
/// KV store and the model) when the launches run.
pub(crate) unsafe fn run(
    gpu: &crate::Gpu,
    launches: &[Launch],
    layout: &Layout,
    a: &mut RunArgs<'_>,
) -> crate::Result<()> {
    use crate::attention::PlanView;
    use crate::engine_ops::CopyArgs;
    use crate::launch::dptr;
    use crate::model::{Indirect, MtpIndirect};
    let s = gpu.stream().clone();
    let ctx = a.ctx;
    let (cap, stride) = (ctx.cap_rows as u64, u64::from(ctx.depths) + 1);
    s.memset_zeros(&mut a.bufs.readback.slice_mut(0..1))?;
    let rb = dptr(&a.bufs.readback, &s);
    let status = rb;
    let (out, counts, plain) = (
        rb + 4,
        rb + 4 * (1 + cap * stride),
        rb + 4 * (1 + cap * stride + cap),
    );
    let table = a.table;
    let word = |off: usize| table + 4 * off as u64;
    let hidden = a.model.config().hidden_size;
    let n = u64::from(a.sampleable);
    let probs = dptr(&a.bufs.probs, &s);
    let view = |p: &PlanAt| PlanView {
        tile: p.tile,
        page_size: p.page_size,
        group_size: p.group_size,
        work_items: p.work_items,
        num_requests: p.num_requests,
        q_indptr: word(p.arrays[0]),
        indices: word(p.arrays[1]),
        indptr: word(p.arrays[2]),
        last_page_len: word(p.arrays[3]),
        request_indices: word(p.arrays[4]),
        qo_tile_indices: word(p.arrays[5]),
        kv_tile_indices: word(p.arrays[6]),
        kv_chunk_size: word(layout.kv_chunk),
        valid: 0,
    };
    for l in launches {
        match l {
            Launch::Copy {
                hidden: wide,
                items,
                arrays,
            } => {
                let width = if *wide { hidden } else { 1 };
                let rows = |words: usize| (words / width) as u64;
                let [normed, prenorm, input] = a.model.level_buffers(&s);
                let state = a.kv.state();
                let taps = a.kv.taps(a.drafter);
                let mut args = CopyArgs {
                    src_buf: word(arrays[0]),
                    src_row: word(arrays[1]),
                    dst_buf: word(arrays[2]),
                    dst_row: word(arrays[3]),
                    status,
                    width: crate::narrow(width, "copy width")?,
                    items: *items,
                    ..CopyArgs::default()
                };
                let table_rows = rows(a.table_words);
                for (b, (base, n)) in [
                    (buf::NORMED, (normed.0, rows(normed.1 * hidden))),
                    (buf::PRENORM, (prenorm.0, rows(prenorm.1 * hidden))),
                    (buf::INPUT, (input.0, rows(input.1 * hidden))),
                    (
                        buf::LEVELS,
                        (dptr(&a.bufs.levels, &s), rows(a.bufs.levels.len())),
                    ),
                    (buf::STATE, (dptr(state, &s), rows(state.len()))),
                    (
                        buf::TAPS,
                        taps.map_or((0, 0), |t| (dptr(t, &s), rows(t.len()))),
                    ),
                    (buf::TABLE, (table, table_rows)),
                    (
                        buf::TARGET,
                        (dptr(&a.bufs.target, &s), rows(a.bufs.target.len())),
                    ),
                ] {
                    args.buf[b as usize] = base;
                    args.rows[b as usize] = n;
                }
                // SAFETY: the item arrays hold `items` words each (the
                // packed table); every buffer's rows are its own, and the
                // kernel bounds every index against them.
                unsafe { a.model.kernels.ops.copy_rows(gpu, args)? };
            }
            Launch::Target {
                tokens,
                token_ids,
                positions,
                kv_block,
                kv_slot,
                plans,
                logit_rows,
                num_logit_rows,
                final_norm,
            } => {
                let src = Indirect {
                    tokens: *tokens as usize,
                    token_ids: word(*token_ids),
                    positions: word(*positions),
                    kv_block: kv_block.iter().map(|&o| word(o)).collect(),
                    kv_slot: kv_slot.iter().map(|&o| word(o)).collect(),
                    plans: plans.iter().map(view).collect(),
                    logit_rows: word(*logit_rows),
                    num_logit_rows: *num_logit_rows as usize,
                    final_norm: *final_norm,
                };
                // Padding rows of every M-padded activation start zero, as on
                // every other path (they reach only padding outputs).
                a.model.zero_padding_rows(gpu, *tokens as usize)?;
                // SAFETY: the caller's contract: the table holds this
                // step's checked token ids (host tokens checked, drafts
                // bounded by the sampler), positions, KV targets, plans and
                // logit rows.
                unsafe { a.model.launch(gpu, a.kv, &src, false)? };
            }
            Launch::Mtp {
                depth,
                tokens,
                token_ids,
                positions,
                kv_rows,
                plan,
                logit_rows,
                num_logit_rows,
            } => {
                let src = MtpIndirect {
                    tokens: *tokens as usize,
                    token_ids: word(*token_ids),
                    positions: word(*positions),
                    kv_rows: word(*kv_rows),
                    plan: view(plan),
                    logit_rows: word(*logit_rows),
                    num_logit_rows: *num_logit_rows as usize,
                    status,
                };
                // SAFETY: as above; the copy before it filled the input rows.
                unsafe { a.model.launch_mtp(gpu, a.kv, *depth as usize, &src)? };
            }
            Launch::Sample {
                rows,
                num_rows,
                draw,
                probs_row,
                tokens,
            } => {
                let launch = crate::sampler::SampleLaunch {
                    logits: dptr(a.model.logits(), &s),
                    logits_stride: u64::from(a.vocab),
                    n: a.sampleable,
                    rows: word(*rows),
                    num_rows: *num_rows,
                    draw: *draw,
                    probs: probs + 8 * u64::from(*probs_row) * n,
                    tokens: match tokens {
                        TokenOut::Lane(i) => word(layout.lane(ctx, *i, 0)),
                        TokenOut::Plain | TokenOut::None => plain,
                    },
                    status,
                };
                // SAFETY: every row's logit row lies inside the last
                // forward's logits; the distributions and tokens it writes
                // lie inside the sampler scratch, a lane or the read-back.
                unsafe { a.model.kernels.sampler.launch_sample(gpu, launch)? };
            }
            Launch::Accept {
                rows,
                target_row,
                draft_row,
                num_drafts,
                num_rows,
            } => {
                let launch = crate::sampler::AcceptLaunch {
                    target: probs,
                    draft: probs,
                    n: a.sampleable,
                    rows: word(*rows),
                    target_row: word(*target_row),
                    draft_row: word(*draft_row),
                    draft_step: ctx.cap_rows,
                    num_drafts: word(*num_drafts),
                    drafts: word(layout.lanes),
                    stride: ctx.depths + 1,
                    scratch: probs
                        + 8 * u64::from(probs_rows(ctx.cap_rows, ctx.depths) - ctx.cap_rows) * n,
                    out,
                    counts,
                    status,
                    num_rows: *num_rows,
                };
                // SAFETY: target rows lie in the target region, draft rows
                // `rd + i × cap` in the draft region and the lanes, every
                // output inside the read-back, as the plan made them.
                unsafe { a.model.kernels.sampler.launch_accept(gpu, launch)? };
            }
        }
    }
    Ok(())
}

/// A padding row of a graph step of draft width `width`: a decode row at
/// position `depths` (so every depth has its rows, as in a uniform real
/// row), drafting `width`, every write and read in the pad blocks, every
/// store into a write-only row, its samples greedy over logits row 0.
pub(crate) fn pad_row(depths: u32, width: u32) -> Row {
    Row {
        slot: None,
        c: depths,
        tokens: vec![crate::graph::PAD_TOKEN],
        k: width,
        sample: true,
        sampling: SamplingParams::greedy(),
        load: Load::Zero,
    }
}

/// KV addressing for the planner over a step's staged tables: real rows
/// through the mirror (its contract checks: a read of an unmapped block or
/// a write into a shared one panics), padding rows into the pad blocks.
pub(crate) struct StepKv<'a> {
    pub mirror: &'a crate::kv::TableMirror,
    pub geometry: &'a [crate::kv::GroupGeometry],
    /// The drafter group's index (the target groups precede it).
    pub drafter: usize,
}

impl KvMap for StepKv<'_> {
    fn target(&self, slot: Option<u32>, group: usize, pos: u32) -> (u32, u32) {
        let g = &self.geometry[group];
        let block = match slot {
            Some(s) => self.mirror.writable_block(s, group, pos),
            None => g.pad_block(),
        };
        (block, pos % g.block_size)
    }

    fn pages(
        &self,
        slot: Option<u32>,
        group: usize,
        pages: std::ops::RangeInclusive<u32>,
    ) -> Vec<u32> {
        let g = &self.geometry[group];
        pages
            .map(|i| match slot {
                Some(s) => self.mirror.block_of(s, group, i * g.block_size),
                None => g.pad_block(),
            })
            .collect()
    }

    fn drafter_row(&self, slot: Option<u32>, pos: u32, write: bool) -> u32 {
        let g = &self.geometry[self.drafter];
        let block = match slot {
            Some(s) if write => self.mirror.writable_block(s, self.drafter, pos),
            Some(s) => self.mirror.block_of(s, self.drafter, pos),
            None => g.pad_block(),
        };
        block * g.block_size + pos % g.block_size
    }

    fn tap_block(&self, slot: Option<u32>, pos: u32) -> u32 {
        match slot {
            Some(s) => self.mirror.writable_block(s, self.drafter, pos),
            None => self.geometry[self.drafter].pad_block(),
        }
    }
}

/// Every forward input `plan` holds, checked as the eager path checks its
/// own before anything takes effect: row counts within the scratch, host
/// token ids inside the vocabulary (drafts are bounded on the device),
/// positions inside the RoPE tables, KV targets and pages inside their
/// pools (pad blocks included, never the null block), logit rows inside
/// each forward and every sample row's logits inside the forward before it.
pub(crate) fn check_plan(
    plan: &Plan,
    model: &crate::model::GpuModel,
    kv: &crate::kv::KvStore,
    drafter: usize,
) -> crate::Result<()> {
    use eidola_engine::spec::NULL_BLOCK;
    let bad = |what: String| Err(crate::CudaError::new(format!("drafted step: {what}")));
    let vocab = model.config().vocab_size;
    let (max_tokens, max_logits, max_len) = (
        model.max_tokens(),
        model.max_logit_rows(),
        model.max_positions(),
    );
    let geometry = kv.geometry();
    let mut logits = 0usize;
    let common = |tokens: &[u32], positions: &[u32], logit_rows: &[u32]| -> crate::Result<()> {
        if tokens.len() > max_tokens || logit_rows.len() > max_logits {
            return bad(format!("{} rows past the scratch", tokens.len()));
        }
        if let Some(t) = tokens.iter().find(|&&t| t as usize >= vocab) {
            return bad(format!("token {t} outside the vocabulary"));
        }
        if let Some(p) = positions.iter().find(|&&p| p as usize >= max_len) {
            return bad(format!("position {p} past the RoPE tables"));
        }
        if let Some(r) = logit_rows.iter().find(|&&r| r as usize >= tokens.len()) {
            return bad(format!("logit row {r} of {}", tokens.len()));
        }
        Ok(())
    };
    for op in &plan.ops {
        match op {
            Op::Target {
                tokens,
                positions,
                kv_targets,
                requests,
                logit_rows,
                ..
            } => {
                common(tokens, positions, logit_rows)?;
                for (g, (targets, reqs)) in kv_targets.iter().zip(requests).enumerate() {
                    let geom = &geometry[g];
                    let ok = |b: u32| b != NULL_BLOCK && (b as usize) < geom.pool_blocks();
                    if let Some(&(b, o)) = targets
                        .iter()
                        .find(|&&(b, o)| !ok(b) || o >= geom.block_size)
                    {
                        return bad(format!("group {g}: KV target block {b} offset {o}"));
                    }
                    if let Some(&b) = reqs.iter().flat_map(|r| &r.pages).find(|&&b| !ok(b)) {
                        return bad(format!("group {g}: page {b}"));
                    }
                }
                logits = logit_rows.len();
            }
            Op::Mtp {
                tokens,
                positions,
                kv_rows,
                requests,
                logit_rows,
                ..
            } => {
                common(tokens, positions, logit_rows)?;
                let geom = &geometry[drafter];
                let rows = geom.pool_rows();
                let bs = geom.block_size as usize;
                let ok = |r: u32| (r as usize) >= bs && (r as usize) < rows;
                if let Some(r) = kv_rows
                    .iter()
                    .chain(requests.iter().flat_map(|r| &r.pages))
                    .find(|&&r| !ok(r))
                {
                    return bad(format!("drafter row {r}"));
                }
                if !logit_rows.is_empty() {
                    logits = logit_rows.len();
                }
            }
            Op::Sample { rows, .. } => {
                if let Some(r) = rows.iter().find(|r| r.logit_row as usize >= logits) {
                    return bad(format!(
                        "sample row reads logits row {} of {logits}",
                        r.logit_row
                    ));
                }
            }
            Op::Copy { .. } | Op::Accept { .. } => {}
        }
    }
    Ok(())
}

/// The rungs of drafted decode steps of draft width `width`: like
/// [`crate::graph::decode_ladder`] over rows that each cost `1 + width`
/// tokens (`min(max_seqs, max_tokens / (1 + width))` per bucket), with the
/// masked expert layout's limit (`128 / (1 + width)` rows) a rung whenever
/// the capacity exceeds it, so a padded step takes the layout its eager run
/// would.
pub fn draft_ladder(buckets: &[eidola_engine::spec::Bucket], width: u32) -> Vec<u32> {
    let rows = |b: &eidola_engine::spec::Bucket| b.max_seqs.min(b.max_tokens / (width + 1));
    let cap = buckets.last().map_or(0, rows);
    let mut rungs: Vec<u32> = (0..32)
        .map(|i| 1u32 << i)
        .take_while(|&r| r < cap)
        .collect();
    rungs.extend(buckets.iter().map(rows).filter(|&r| r > 0));
    let masked = crate::model::MASKED_TOKENS / (width + 1);
    if masked > 0 && masked < cap {
        rungs.push(masked);
    }
    rungs.sort_unstable();
    rungs.dedup();
    rungs
}

/// The drafted ladders an executor of `depths` draft depths captures: one
/// per width `0 ..= depths`, since the serving core narrows a step's width to
/// keep it within the masked layout (`ModelSpec::draft_step_tokens`).
pub fn draft_ladders(buckets: &[eidola_engine::spec::Bucket], depths: u32) -> Vec<(u32, Vec<u32>)> {
    (0..=depths)
        .map(|w| (w, draft_ladder(buckets, w)))
        .collect()
}

struct RungGraph {
    width: u32,
    rows: usize,
    layout: Layout,
    launches: Vec<Launch>,
    graph: cudarc::driver::CudaGraph,
}

/// The captured drafted-step graphs: one per rung of each width's
/// [`draft_ladder`], over one step table sized for the largest rung's
/// layout.
pub struct DraftGraphs {
    rungs: Vec<RungGraph>,
    table: cudarc::driver::CudaSlice<u32>,
    capture_bytes: Vec<i64>,
}

impl std::fmt::Debug for DraftGraphs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DraftGraphs")
            .field("rungs", &self.rungs())
            .finish_non_exhaustive()
    }
}

impl DraftGraphs {
    /// For each width and each rung of its ladder, in order: plan an
    /// all-padding step, run its launches once directly (which exercises
    /// every launch on this device), capture them, upload and replay the
    /// capture once, checking the status word after each run.
    pub(crate) fn capture(
        gpu: &crate::Gpu,
        ladders: Vec<(u32, Vec<u32>)>,
        ctx: &PlanCtx,
        t: &mut RunTarget<'_>,
    ) -> crate::Result<DraftGraphs> {
        use cudarc::driver::sys;
        let mut ctx = ctx.clone();
        ctx.graph = true;
        let mut planned = Vec::new();
        for (width, r) in ladders
            .iter()
            .flat_map(|(w, l)| l.iter().map(move |&r| (*w, r)))
        {
            let rows: Vec<Row> = (0..r).map(|_| pad_row(ctx.depths, width)).collect();
            let kv = StepKv {
                mirror: t.kv.mirror(),
                geometry: t.kv.geometry(),
                drafter: t.drafter,
            };
            let plan = plan(&rows, &ctx, &kv);
            check_plan(&plan, t.model, t.kv, t.drafter)?;
            let built = build(&plan, &ctx)?;
            let layout = Layout::new(&built, &ctx, true)?;
            let words = pack(&built, &layout, &ctx)?;
            let launches = launches(&built, &layout);
            planned.push((width, r as usize, layout, launches, words));
        }
        let max_words = planned.iter().map(|p| p.2.words).max().unwrap_or(1);
        let s = gpu.stream().clone();
        let mut table = s.alloc_zeros::<u32>(max_words.max(1))?;
        let mut rungs = Vec::with_capacity(planned.len());
        let mut capture_bytes = Vec::with_capacity(planned.len());
        for (width, rows, layout, launches, words) in planned {
            s.memcpy_htod(&words, &mut table.slice_mut(..words.len()))?;
            let base = crate::launch::dptr(&table, &s);
            let check = |t: &RunTarget<'_>| -> crate::Result<()> {
                let status = s.clone_dtoh(&t.bufs.readback.slice(0..1))?[0];
                if status != 0 {
                    return Err(crate::CudaError::new(format!(
                        "drafted rung {rows} of width {width}: status {status:#x} over padding"
                    )));
                }
                Ok(())
            };
            // SAFETY: the table holds the packed all-padding step.
            unsafe { run(gpu, &launches, &layout, &mut t.args(&ctx, base, max_words))? };
            check(t)?;
            let before = crate::graph::free_memory()?;
            s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            // SAFETY: as above; nothing runs while capturing.
            let recorded =
                unsafe { run(gpu, &launches, &layout, &mut t.args(&ctx, base, max_words)) };
            let graph = s.end_capture(sys::CUgraphInstantiate_flags(0));
            recorded?;
            let graph = graph?.ok_or_else(|| {
                crate::CudaError::new(format!(
                    "drafted rung {rows} of width {width}: the capture is empty"
                ))
            })?;
            graph.upload()?;
            graph.launch()?;
            check(t)?;
            capture_bytes.push(before - crate::graph::free_memory()?);
            rungs.push(RungGraph {
                width,
                rows,
                layout,
                launches,
                graph,
            });
        }
        Ok(DraftGraphs {
            rungs,
            table,
            capture_bytes,
        })
    }

    /// Every rung as `(width, rows)`, in capture order (by width, then rows
    /// ascending).
    pub fn rungs(&self) -> Vec<(u32, u32)> {
        self.rungs
            .iter()
            .map(|r| (r.width, u32::try_from(r.rows).expect("rows fit u32")))
            .collect()
    }

    /// Row counts of width `width`'s rungs, ascending.
    pub fn ladder(&self, width: u32) -> Vec<u32> {
        self.rungs()
            .into_iter()
            .filter(|&(w, _)| w == width)
            .map(|(_, r)| r)
            .collect()
    }

    /// Device memory each capture took, as the driver's free memory moved
    /// (in the order of [`DraftGraphs::rungs`]).
    pub fn capture_bytes(&self) -> &[i64] {
        &self.capture_bytes
    }

    /// The smallest rung of width `width` holding `rows` rows.
    pub(crate) fn rung_for(&self, width: u32, rows: usize) -> Option<usize> {
        if rows == 0 {
            return None;
        }
        self.rungs
            .iter()
            .position(|r| r.width == width && r.rows >= rows)
    }

    pub(crate) fn rows(&self, rung: usize) -> usize {
        self.rungs[rung].rows
    }

    pub(crate) fn layout(&self, rung: usize) -> &Layout {
        &self.rungs[rung].layout
    }

    pub(crate) fn launches(&self, rung: usize) -> &[Launch] {
        &self.rungs[rung].launches
    }

    pub(crate) fn upload(&mut self, gpu: &crate::Gpu, words: &[u32]) -> crate::Result<()> {
        if words.len() > self.table.len() {
            return Err(crate::CudaError::new("drafted step table overflows"));
        }
        gpu.stream()
            .memcpy_htod(words, &mut self.table.slice_mut(..words.len()))?;
        Ok(())
    }

    pub(crate) fn table_ptr(&self, s: &cudarc::driver::CudaStream) -> u64 {
        crate::launch::dptr(&self.table, s)
    }

    pub(crate) fn table_words(&self) -> usize {
        self.table.len()
    }

    pub(crate) fn replay(&self, rung: usize) -> crate::Result<()> {
        Ok(self.rungs[rung].graph.launch()?)
    }

    pub(crate) fn read_words(
        &self,
        gpu: &crate::Gpu,
        range: std::ops::Range<usize>,
    ) -> crate::Result<Vec<u32>> {
        Ok(gpu.stream().clone_dtoh(&self.table.slice(range))?)
    }
}

#[cfg(test)]
mod tests {
    //! The planner, layout, packing and launch list, run on the host by an
    //! emulator that executes a packed table's launches symbolically: every
    //! hidden row is a tag naming whose state at which position it holds
    //! (the target's, or an MTP depth's output), every token a host token or
    //! a named draft. The emulator checks each forward's inputs (the
    //! target's state at the anchor `s - 1 - d`, the token at `s`, the RoPE
    //! position, the KV row, the attention pages)
    //! and the state and taps the step leaves, against the definition in the
    //! module docs.

    use super::*;
    use std::collections::{HashMap, HashSet};

    /// A hidden row's contents.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum V {
        Zero,
        /// The target's state of row `row` at position `pos`.
        H {
            row: usize,
            pos: u32,
        },
        /// MTP depth `depth`'s output for row `row` at slot `pos` (feeds
        /// nothing but its logits).
        M {
            row: usize,
            depth: u32,
            pos: u32,
        },
        /// A padding row's level (never checked against anything real).
        Pad,
    }

    /// A token id: host tokens are small integers, drafts are tagged.
    fn draft_token(row: usize, j: u32) -> u32 {
        1_000_000 + 100 * u32::try_from(row).unwrap() + j
    }

    const PAD_BLOCK: u32 = 9_999;

    /// Blocks are unique per (row, group, logical block); group
    /// `targets.len()` is the drafter.
    struct FakeKv {
        bs: u32,
    }

    impl FakeKv {
        fn block(&self, slot: Option<u32>, group: usize, logical: u32) -> u32 {
            match slot {
                None => PAD_BLOCK,
                Some(s) => 1 + s * 1000 + u32::try_from(group).unwrap() * 100 + logical,
            }
        }
    }

    impl KvMap for FakeKv {
        fn target(&self, slot: Option<u32>, group: usize, pos: u32) -> (u32, u32) {
            (self.block(slot, group, pos / self.bs), pos % self.bs)
        }
        fn pages(
            &self,
            slot: Option<u32>,
            group: usize,
            pages: std::ops::RangeInclusive<u32>,
        ) -> Vec<u32> {
            pages.map(|i| self.block(slot, group, i)).collect()
        }
        fn drafter_row(&self, slot: Option<u32>, pos: u32, _write: bool) -> u32 {
            self.block(slot, 7, pos / self.bs) * self.bs + pos % self.bs
        }
        fn tap_block(&self, slot: Option<u32>, pos: u32) -> u32 {
            self.block(slot, 7, pos / self.bs)
        }
    }

    fn ctx(depths: u32, bs: u32, cap_rows: u32, graph: bool) -> PlanCtx {
        PlanCtx {
            depths,
            hidden: MtpHidden::Normed,
            block_size: bs,
            targets: vec![
                TargetGroup {
                    attention: AttentionKind::Full,
                    group_size: 16,
                },
                TargetGroup {
                    attention: AttentionKind::Sliding { window: 8 },
                    group_size: 8,
                },
            ],
            drafter_window: 8,
            drafter_group_size: 8,
            cap_rows,
            max_blocks: 64,
            graph,
        }
    }

    /// The emulated device state after a step.
    struct World {
        /// Hidden rows by (buffer, row).
        rows: HashMap<(u32, u32), V>,
        table: Vec<u32>,
        /// Drafter KV writes: (row, depth, position) -> count.
        drafter_kv: HashMap<(usize, u32, u32), u32>,
        /// Target KV writes: (group, block, offset) -> (row, position).
        target_kv: HashMap<(usize, u32, u32), (usize, u32)>,
        /// The last head's logit rows: per row of the logits, (row,
        /// position, producer) with producer `None` for the target and
        /// `Some(depth)` for an MTP depth.
        logits: Vec<(usize, u32, Option<u32>)>,
        /// Drafted rows' outputs: accepted counts are not modelled; the
        /// drafts each lane holds are.
        lanes_written: HashSet<(u32, u32)>,
        accepts: u32,
        plain_samples: u32,
    }

    /// Which row (by index) each table row's item belongs to, for a forward:
    /// recovered from the planner's own op (the emulator checks the table
    /// against it rather than trusting it).
    fn emulate(rows: &[Row], ctx: &PlanCtx, kv: &FakeKv) -> (Plan, World) {
        let plan = plan(rows, ctx, kv);
        let built = build(&plan, ctx).unwrap();
        let layout = Layout::new(&built, ctx, ctx.graph).unwrap();
        let table = pack(&built, &layout, ctx).unwrap();
        let launches = launches(&built, &layout);
        let d_count = ctx.depths;
        let mut w = World {
            rows: HashMap::new(),
            table,
            drafter_kv: HashMap::new(),
            target_kv: HashMap::new(),
            logits: Vec::new(),
            lanes_written: HashSet::new(),
            accepts: 0,
            plain_samples: 0,
        };
        // Seed what loads read: each real row's levels at c - 1 (level `l`
        // the target's state at `c - 1 - l`).
        w.rows.insert((buf::LEVELS, Levels::ZERO), V::Zero);
        for (r, row) in rows.iter().enumerate() {
            for l in 0..d_count {
                let v = if row.c > l {
                    V::H {
                        row: r,
                        pos: row.c - 1 - l,
                    }
                } else {
                    V::Zero
                };
                match row.load {
                    Load::State { entry } => {
                        w.rows.insert(
                            (buf::STATE, state_row(d_count, row.slot.unwrap(), entry, l)),
                            v,
                        );
                    }
                    Load::Tap { block } => {
                        w.rows.insert((buf::TAPS, tap_row(d_count, block, l)), v);
                    }
                    Load::None | Load::Zero => {}
                }
            }
        }
        let word = |w: &World, off: usize| w.table[off];
        let words = |w: &World, off: usize, n: usize| w.table[off..off + n].to_vec();
        // Which (row, position) each forward row is: recovered from the
        // positions and KV rows the table holds, matched against the rows.
        let mut level_buf: HashMap<u32, V> = HashMap::new();
        let level_id = match ctx.hidden {
            MtpHidden::Normed => buf::NORMED,
            MtpHidden::PreNorm => buf::PRENORM,
        };
        let row_of_slot = |slot: Option<u32>| -> Option<usize> { slot.map(|s| s as usize) };
        let _ = row_of_slot;
        let mut lanes: HashMap<(u32, u32), u32> = HashMap::new();
        let mut op_rows: Vec<(usize, u32)> = Vec::new();
        for (li, l) in launches.iter().enumerate() {
            match l {
                Launch::Copy {
                    hidden,
                    items,
                    arrays,
                } => {
                    let n = *items as usize;
                    let sb = words(&w, arrays[0], n);
                    let sr = words(&w, arrays[1], n);
                    let db = words(&w, arrays[2], n);
                    let dr = words(&w, arrays[3], n);
                    // No item's destination is another's source or
                    // destination.
                    let srcs: HashSet<(u32, u32)> =
                        sb.iter().copied().zip(sr.iter().copied()).collect();
                    let mut dsts = HashSet::new();
                    for (&b, &r) in db.iter().zip(&dr) {
                        assert!(
                            dsts.insert((b, r)),
                            "launch {li}: destination {b}/{r} twice"
                        );
                        assert!(
                            !srcs.contains(&(b, r)),
                            "launch {li}: {b}/{r} read and written"
                        );
                    }
                    if *hidden {
                        let mut writes = Vec::new();
                        for i in 0..n {
                            let src = match sb[i] {
                                b if b == level_id => *level_buf.get(&sr[i]).unwrap_or_else(|| {
                                    panic!("launch {li}: level row {} unset", sr[i])
                                }),
                                b => *w.rows.get(&(b, sr[i])).unwrap_or_else(|| {
                                    panic!("launch {li}: item {i} reads unset {b}/{}", sr[i])
                                }),
                            };
                            assert_ne!(db[i], buf::TABLE, "a hidden copy into the table");
                            writes.push(((db[i], dr[i]), src));
                        }
                        for (k, v) in writes {
                            w.rows.insert(k, v);
                        }
                    } else {
                        for i in 0..n {
                            assert_eq!((sb[i], db[i]), (buf::TABLE, buf::TABLE));
                            let lane_word = sr[i] as usize;
                            assert!(lane_word >= layout.lanes && lane_word < layout.kv_chunk);
                            let rel = u32::try_from(lane_word - layout.lanes).unwrap();
                            let (lane, rd) = (rel / ctx.cap_rows, rel % ctx.cap_rows);
                            assert!(
                                lanes.contains_key(&(lane, rd)),
                                "launch {li}: draft lane {lane} of row {rd} copied before it was drawn"
                            );
                            w.table[dr[i] as usize] = lanes[&(lane, rd)];
                        }
                    }
                }
                Launch::Mtp {
                    depth,
                    tokens,
                    token_ids,
                    positions,
                    kv_rows,
                    plan,
                    logit_rows,
                    num_logit_rows,
                } => {
                    let t = *tokens as usize;
                    let toks = words(&w, *token_ids, t);
                    let pos = words(&w, *positions, t);
                    let kvr = words(&w, *kv_rows, t);
                    let mut outs = HashMap::new();
                    op_rows.clear();
                    for n in 0..t {
                        // Which row: the one whose drafter rows hold this KV
                        // row (padding rows share the pad block), at the
                        // slot its RoPE position (the anchor) implies.
                        let s = pos[n] + 1 + depth;
                        let (r, row) = rows
                            .iter()
                            .enumerate()
                            .find(|(_, row)| {
                                kv.drafter_row(row.slot, s, true) == kvr[n]
                                    && s >= row.c
                                    && s <= row.p() + row.k
                                    && (row.slot.is_some() || true)
                            })
                            .expect("an MTP row belongs to a step row");
                        op_rows.push((r, s));
                        let input = w.rows[&(buf::INPUT, u32::try_from(n).unwrap())];
                        if row.slot.is_some() {
                            assert!(s > *depth, "depth {depth} row at {s}");
                            let want = V::H {
                                row: r,
                                pos: s - 1 - depth,
                            };
                            assert_eq!(input, want, "depth {depth} row {r} at {s}: input");
                            let p = row.p();
                            let tok = if s <= p {
                                row.tokens[(s - row.c) as usize]
                            } else {
                                draft_token(r, s - p)
                            };
                            assert_eq!(toks[n], tok, "depth {depth} row {r} at {s}: token");
                            *w.drafter_kv.entry((r, *depth, s)).or_default() += 1;
                        }
                        outs.insert(
                            u32::try_from(n).unwrap(),
                            if row.slot.is_some() {
                                V::M {
                                    row: r,
                                    depth: *depth,
                                    pos: s,
                                }
                            } else {
                                V::Pad
                            },
                        );
                    }
                    check_mtp_pages(&w, plan, &op_rows, rows, kv, *depth, ctx);
                    level_buf = outs;
                    // The head over the listed rows.
                    let lr = words(&w, *logit_rows, *num_logit_rows as usize);
                    w.logits = lr
                        .iter()
                        .map(|&i| {
                            let (r, s) = op_rows[i as usize];
                            (r, s, Some(*depth))
                        })
                        .collect();
                }
                Launch::Target {
                    tokens,
                    token_ids,
                    positions,
                    kv_block,
                    kv_slot,
                    plans,
                    logit_rows,
                    num_logit_rows,
                    final_norm,
                } => {
                    assert!(*final_norm);
                    let t = *tokens as usize;
                    let toks = words(&w, *token_ids, t);
                    let pos = words(&w, *positions, t);
                    let mut outs = HashMap::new();
                    op_rows.clear();
                    let mut n = 0usize;
                    for (r, row) in rows.iter().enumerate() {
                        let p = row.p();
                        for x in row.c..=p + row.k {
                            assert_eq!(pos[n], x, "target row {r}");
                            let tok = if x <= p {
                                row.tokens[(x - row.c) as usize]
                            } else if row.slot.is_some() {
                                draft_token(r, x - p)
                            } else {
                                toks[n]
                            };
                            assert_eq!(toks[n], tok, "target row {r} at {x}: token");
                            for g in 0..ctx.targets.len() {
                                let b = word(&w, kv_block[g] + n);
                                let o = word(&w, kv_slot[g] + n);
                                assert_eq!((b, o), kv.target(row.slot, g, x));
                                if row.slot.is_some() {
                                    assert!(
                                        w.target_kv.insert((g, b, o), (r, x)).is_none(),
                                        "target KV written twice"
                                    );
                                }
                            }
                            outs.insert(
                                u32::try_from(n).unwrap(),
                                if row.slot.is_some() {
                                    V::H { row: r, pos: x }
                                } else {
                                    V::Pad
                                },
                            );
                            op_rows.push((r, x));
                            n += 1;
                        }
                    }
                    assert_eq!(n, t);
                    for (g, plan) in plans.iter().enumerate() {
                        check_target_pages(&w, plan, rows, kv, g, ctx);
                    }
                    level_buf = outs;
                    let lr = words(&w, *logit_rows, *num_logit_rows as usize);
                    w.logits = lr
                        .iter()
                        .map(|&i| {
                            let (r, x) = op_rows[i as usize];
                            (r, x, None)
                        })
                        .collect();
                }
                Launch::Sample {
                    rows: at,
                    num_rows,
                    draw,
                    probs_row,
                    tokens,
                } => {
                    for i in 0..*num_rows {
                        let sr = words(&w, at + 8 * i as usize, 8);
                        let logit_row = sr[7] as usize;
                        assert!(
                            logit_row < w.logits.len(),
                            "sample row reads logits row {logit_row}"
                        );
                        let position = sr[6];
                        match tokens {
                            TokenOut::Lane(lane) => {
                                assert_eq!(*draw, Some(Stream::Draft));
                                assert_eq!(*probs_row, lane * ctx.cap_rows);
                                // Drafting row `i`'s draft `lane + 1`: the
                                // depth-`lane` prediction at `p + lane`.
                                let (r, row) = rows
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, row)| row.k > 0)
                                    .nth(i as usize)
                                    .unwrap();
                                if row.slot.is_some() && *lane < row.k {
                                    assert_eq!(
                                        w.logits[logit_row],
                                        (r, row.p() + lane, Some(*lane)),
                                        "draft {} of row {r}",
                                        lane + 1
                                    );
                                    assert_eq!(position, row.p() + 1 + lane);
                                    lanes.insert((*lane, i), draft_token(r, lane + 1));
                                } else {
                                    // Unused: a token of the vocabulary.
                                    lanes.insert((*lane, i), 0);
                                }
                                w.lanes_written.insert((*lane, i));
                            }
                            TokenOut::None => assert_eq!(*draw, None),
                            TokenOut::Plain => {
                                assert_eq!(*draw, Some(Stream::Sample));
                                w.plain_samples += 1;
                            }
                        }
                    }
                }
                Launch::Accept {
                    rows: at,
                    target_row,
                    draft_row,
                    num_drafts,
                    num_rows,
                } => {
                    let tr = words(&w, *target_row, *num_rows as usize);
                    let dr = words(&w, *draft_row, *num_rows as usize);
                    let nd = words(&w, *num_drafts, *num_rows as usize);
                    for (i, (r, row)) in rows
                        .iter()
                        .enumerate()
                        .filter(|(_, row)| row.k > 0)
                        .enumerate()
                    {
                        assert_eq!(dr[i], u32::try_from(i).unwrap());
                        assert_eq!(nd[i], row.k);
                        let sr = words(&w, at + 8 * i, 8);
                        if row.slot.is_some() {
                            assert_eq!(sr[6], row.p() + 1, "row {r}'s first drafted position");
                            for j in 0..row.k {
                                assert_eq!(lanes[&(j, dr[i])], draft_token(r, j + 1));
                            }
                        }
                        assert!(tr[i] >= ctx.depths * ctx.cap_rows);
                        w.accepts += 1;
                    }
                }
            }
        }
        (plan, w)
    }

    /// Every query of an MTP launch sees exactly the positions from the
    /// first its window and its depth allow to itself, in order.
    fn check_mtp_pages(
        w: &World,
        p: &PlanAt,
        op_rows: &[(usize, u32)],
        rows: &[Row],
        kv: &FakeKv,
        depth: u32,
        ctx: &PlanCtx,
    ) {
        assert_eq!(p.page_size, 1);
        let reqs = p.num_requests as usize;
        let qi = &w.table[p.arrays[0]..p.arrays[0] + reqs + 1];
        let ip = &w.table[p.arrays[2]..p.arrays[2] + reqs + 1];
        for q in 0..reqs {
            let (q0, q1) = (qi[q] as usize, qi[q + 1] as usize);
            let pages = &w.table[p.arrays[1] + ip[q] as usize..p.arrays[1] + ip[q + 1] as usize];
            let (r, first) = op_rows[q0];
            let (_, last) = op_rows[q1 - 1];
            assert!(op_rows[q0..q1].iter().all(|&(rr, _)| rr == r));
            assert_eq!(q1 - q0, (last - first + 1) as usize);
            let row = &rows[r];
            let window = AttentionKind::Sliding {
                window: ctx.drafter_window,
            };
            let lo = window.first_visible(first).max(depth + 1);
            let want: Vec<u32> = (lo..=last)
                .map(|x| kv.drafter_row(row.slot, x, false))
                .collect();
            assert_eq!(
                pages,
                &want[..],
                "depth {depth} row {r} queries {first}..={last}"
            );
            // Every position listed has this depth's KV (written this step
            // or before it).
            if row.slot.is_some() {
                for x in lo..=last {
                    assert!(x > depth);
                }
            }
        }
    }

    fn check_target_pages(
        w: &World,
        p: &PlanAt,
        rows: &[Row],
        kv: &FakeKv,
        g: usize,
        ctx: &PlanCtx,
    ) {
        let reqs = p.num_requests as usize;
        assert_eq!(reqs, rows.len());
        let ip = &w.table[p.arrays[2]..p.arrays[2] + reqs + 1];
        let last_len = &w.table[p.arrays[3]..p.arrays[3] + reqs];
        for (r, row) in rows.iter().enumerate() {
            let pages = &w.table[p.arrays[1] + ip[r] as usize..p.arrays[1] + ip[r + 1] as usize];
            let first_page = ctx.targets[g].attention.first_visible(row.c) / ctx.block_size;
            let last = row.p() + row.k;
            assert_eq!(
                pages,
                &kv.pages(row.slot, g, first_page..=last / ctx.block_size)[..]
            );
            let kv_len = last + 1 - first_page * ctx.block_size;
            assert_eq!(
                (u32::try_from(pages.len()).unwrap() - 1) * ctx.block_size + last_len[r],
                kv_len
            );
        }
    }

    /// After the step: the state of every real row holds its levels at `p
    /// ..= p + k` (zeros where a level does not exist), every block whose
    /// last position the row covered holds its tap, and every drafter row
    /// that exists was computed exactly once.
    fn check_world(rows: &[Row], ctx: &PlanCtx, kv: &FakeKv, w: &World) {
        let d_count = ctx.depths;
        let bs = ctx.block_size;
        for (r, row) in rows.iter().enumerate() {
            let Some(slot) = row.slot else { continue };
            let p = row.p();
            let lv = |l: u32, x: u32| {
                if x >= l {
                    V::H { row: r, pos: x - l }
                } else {
                    V::Zero
                }
            };
            for x in p..=p + row.k {
                for l in 0..d_count {
                    assert_eq!(
                        w.rows
                            .get(&(buf::STATE, state_row(d_count, slot, x - p, l))),
                        Some(&lv(l, x)),
                        "row {r} state at {x} level {l}"
                    );
                }
            }
            for x in row.c..=p + row.k {
                if (x + 1).is_multiple_of(bs) {
                    for l in 0..d_count {
                        assert_eq!(
                            w.rows
                                .get(&(buf::TAPS, tap_row(d_count, kv.tap_block(row.slot, x), l))),
                            Some(&lv(l, x)),
                            "row {r} tap at {x} level {l}"
                        );
                    }
                }
            }
            for d in 0..d_count {
                let lo = if row.k > 0 { p } else { row.c };
                for s in lo..=p + row.k {
                    let want = u32::from(s > d);
                    assert_eq!(
                        w.drafter_kv.get(&(r, d, s)).copied().unwrap_or(0),
                        want,
                        "row {r} depth {d} at {s}"
                    );
                }
            }
            for x in row.c..=p + row.k {
                for g in 0..ctx.targets.len() {
                    let (b, o) = kv.target(row.slot, g, x);
                    assert_eq!(w.target_kv.get(&(g, b, o)), Some(&(r, x)));
                }
            }
        }
        let drafting = rows.iter().filter(|r| r.k > 0).count();
        assert_eq!(w.accepts as usize, drafting);
        assert_eq!(
            w.plain_samples as usize,
            rows.iter().filter(|r| r.sample && r.k == 0).count()
        );
    }

    fn sampling(r: usize) -> SamplingParams {
        if r.is_multiple_of(2) {
            SamplingParams::greedy()
        } else {
            SamplingParams::new(0.7, 0, 1.0, 0.0, 77 + r as u64).unwrap()
        }
    }

    fn decode(slot: u32, p: u32, k: u32, load: Load) -> Row {
        Row {
            slot: Some(slot),
            c: p,
            tokens: vec![10 + slot],
            k,
            sample: true,
            sampling: sampling(slot as usize),
            load,
        }
    }

    fn prefill(slot: u32, c: u32, n: u32, sample: bool, load: Load) -> Row {
        Row {
            slot: Some(slot),
            c,
            tokens: (0..n).map(|i| 20 + i).collect(),
            k: 0,
            sample,
            sampling: sampling(slot as usize),
            load,
        }
    }

    fn pad(depths: u32) -> Row {
        pad_row(depths, depths)
    }

    /// Mixed steps across depths and block sizes: decode rows at every
    /// draft width (including near the sequence start, where some depths
    /// have no rows yet), prefill chunks from position 0 and from a block
    /// boundary, rows that do not sample, padding rows.
    #[test]
    fn mixed_steps_compute_every_drafter_row_once_from_the_right_inputs() {
        for depths in 1..=3u32 {
            for bs in [1u32, 2, 3, 4, 16] {
                let kv = FakeKv { bs };
                for graph in [false, true] {
                    let mut rows = vec![];
                    let mut slot = 0;
                    for p in 1..=depths + 2 {
                        for k in 0..=depths {
                            rows.push(decode(
                                slot,
                                p,
                                k,
                                Load::State {
                                    entry: k % (depths + 1),
                                },
                            ));
                            slot += 1;
                        }
                    }
                    // A long-context decode row at every alignment.
                    for p in [bs * 5 - 1, bs * 5, bs * 5 + 1, 40] {
                        rows.push(decode(slot, p, depths, Load::State { entry: 0 }));
                        slot += 1;
                    }
                    // Resumed from a tap at a block boundary.
                    rows.push(decode(slot, bs * 3, depths, Load::Tap { block: 4242 }));
                    slot += 1;
                    rows.push(prefill(slot, 0, 13, true, Load::None));
                    slot += 1;
                    rows.push(prefill(slot, 0, 1, false, Load::None));
                    slot += 1;
                    rows.push(prefill(slot, bs * 2, 9, false, Load::Tap { block: 4343 }));
                    slot += 1;
                    rows.push(prefill(slot, 7, 3, true, Load::State { entry: 1 }));
                    rows.push(pad(depths));
                    let cap = u32::try_from(rows.len()).unwrap() + 2;
                    let c = ctx(depths, bs, cap, graph);
                    let (_, w) = emulate(&rows, &c, &kv);
                    check_world(&rows, &c, &kv, &w);
                }
            }
        }
    }

    /// Pre-norm chaining reads the residual stream's rows.
    #[test]
    fn pre_norm_feeds_the_residual_stream() {
        let kv = FakeKv { bs: 4 };
        let mut c = ctx(3, 4, 4, false);
        c.hidden = MtpHidden::PreNorm;
        let rows = vec![
            decode(0, 9, 3, Load::State { entry: 1 }),
            prefill(1, 0, 6, true, Load::None),
        ];
        let p = plan(&rows, &c, &kv);
        let reads: HashSet<u32> = p
            .ops
            .iter()
            .filter_map(|op| match op {
                Op::Copy { items, .. } => Some(items),
                _ => None,
            })
            .flatten()
            .filter_map(|i| match i.src {
                Loc::Buf(b, _) if b == buf::NORMED || b == buf::PRENORM => Some(b),
                _ => None,
            })
            .collect();
        assert_eq!(reads, HashSet::from([buf::PRENORM]));
        assert!(p.ops.iter().all(|op| !matches!(
            op,
            Op::Target {
                final_norm: true,
                ..
            }
        )));
    }

    fn bucket(max_seqs: u32, max_tokens: u32) -> eidola_engine::spec::Bucket {
        eidola_engine::spec::Bucket {
            max_seqs,
            max_tokens,
        }
    }

    /// Rows of a drafted rung cost `1 + D` tokens each, and the padded step
    /// takes the expert layout its eager run would: its target tokens are
    /// at most 128 exactly when the real step's are.
    #[test]
    fn the_drafted_ladder_keeps_the_masked_layout() {
        assert_eq!(
            draft_ladder(&[bucket(64, 8192)], 3),
            vec![1, 2, 4, 8, 16, 32, 64]
        );
        assert_eq!(
            draft_ladder(&[bucket(64, 8192)], 2),
            vec![1, 2, 4, 8, 16, 32, 42, 64]
        );
        // Tokens bound the rows: 100 tokens hold 25 rows of 1 + 3.
        assert_eq!(
            draft_ladder(&[bucket(64, 100)], 3),
            vec![1, 2, 4, 8, 16, 25]
        );
        assert_eq!(draft_ladder(&[bucket(64, 3)], 3), Vec::<u32>::new());
        for depths in 1..=3u32 {
            for cap in 1..=300u32 {
                let ladder = draft_ladder(&[bucket(cap, 8192)], depths);
                assert!(ladder.windows(2).all(|w| w[0] < w[1]));
                let rows_cap = cap.min(8192 / (depths + 1));
                assert_eq!(*ladder.last().unwrap(), rows_cap);
                for rows in 1..=rows_cap {
                    let rung = *ladder.iter().find(|&&r| r >= rows).unwrap();
                    let tokens = |n: u32| n * (depths + 1);
                    assert_eq!(
                        tokens(rows) <= 128,
                        tokens(rung) <= 128,
                        "D {depths} cap {cap}: {rows} rows in rung {rung}"
                    );
                }
            }
        }
    }

    /// Every uniform drafted decode step (each row one host token, sampled,
    /// the full draft width, past position `D`), padded to a rung, has the
    /// launch list of that rung's all-padding capture and fits its table,
    /// whatever its rows' positions (every alignment with the blocks, every
    /// distance into the window), sampling, and where their levels load
    /// from; and it emulates correctly.
    #[test]
    fn padded_uniform_steps_have_their_rungs_shape() {
        for depths in 1..=3u32 {
            for (bs, width) in [1u32, 3, 16]
                .into_iter()
                .flat_map(|bs| (0..=depths).map(move |w| (bs, w)))
            {
                let kv = FakeKv { bs };
                let cap = 12u32;
                let c = ctx(depths, bs, cap, true);
                for &rung in &[1u32, 5, 12] {
                    let capture: Vec<Row> = (0..rung).map(|_| pad_row(depths, width)).collect();
                    let cp = plan(&capture, &c, &kv);
                    let cb = build(&cp, &c).unwrap();
                    let layout = Layout::new(&cb, &c, true).unwrap();
                    let want = launches(&cb, &layout);
                    for real in 1..=rung {
                        let mut rows: Vec<Row> = (0..real)
                            .map(|i| {
                                let p = depths + i * 7 + (i % 3) * bs;
                                let load = if i % 2 == 0 {
                                    Load::State {
                                        entry: i % (depths + 1),
                                    }
                                } else {
                                    Load::Tap { block: 500 + i }
                                };
                                decode(i, p, width, load)
                            })
                            .collect();
                        rows.extend((real..rung).map(|_| pad_row(depths, width)));
                        let p = plan(&rows, &c, &kv);
                        let b = build(&p, &c).unwrap();
                        assert_eq!(
                            launches(&b, &layout),
                            want,
                            "D {depths} width {width} bs {bs} rung {rung} real {real}"
                        );
                        pack(&b, &layout, &c).unwrap();
                        let (_, w) = emulate(&rows, &c, &kv);
                        check_world(&rows, &c, &kv, &w);
                    }
                }
            }
        }
    }

    /// The seam's contract, checked before anything is launched: no drafts
    /// at position 0, no more drafts than depths, drafts only on decode rows
    /// that sample.
    #[test]
    fn draft_contract_violations_panic() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let kv = FakeKv { bs: 16 };
        let c = ctx(2, 16, 4, false);
        let refused = |rows: Vec<Row>| {
            catch_unwind(AssertUnwindSafe(|| plan(&rows, &c, &kv))).expect_err("refused");
        };
        refused(vec![decode(0, 0, 1, Load::None)]);
        refused(vec![decode(0, 5, 3, Load::State { entry: 0 })]);
        let mut two = prefill(0, 4, 2, true, Load::State { entry: 0 });
        two.k = 1;
        refused(vec![two]);
        let mut unsampled = decode(0, 5, 1, Load::State { entry: 0 });
        unsampled.sample = false;
        refused(vec![unsampled]);
        // A row past position 0 must load its levels, and one at 0 must not.
        refused(vec![decode(0, 5, 1, Load::None)]);
        refused(vec![prefill(0, 0, 3, true, Load::Zero)]);
        plan(&[decode(0, 5, 2, Load::State { entry: 0 })], &c, &kv);
    }
}
