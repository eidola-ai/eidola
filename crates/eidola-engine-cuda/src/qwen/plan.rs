//! A step as data: what every launch of the step program ([`super::program`]) reads,
//! laid out in one table of 32-bit words that one host-to-device copy fills.
//!
//! The executor describes a step as a [`StepPlan`]: the drafter's passes that draw the
//! drafts, the one target forward over every row's host tokens and drafts (its DeltaNet
//! sequences, the row copies that stage and keep the recurrent state's inputs, the
//! attention rows and, in the fast profile, FA2 requests), the head's rows, the drafter
//! over the host positions of rows that do not draft, the sample rows and chain
//! acceptance, the canonical rewrite, and the copies that keep the drafter's states. Every
//! count the launches take is a [`Shape`]; [`Layout::new`] places every array at an
//! offset that is a function of the shape alone, and [`pack`] writes a plan into it. An
//! eager step's shape is its plan's own ([`Shape::of`]); a graph rung's is fixed, and a
//! plan is packed into it only if each of its variable lists fits, the rest padded with
//! inert entries (a copy of a pad row onto itself, a fully padded attention request). So
//! a captured launch sequence, which fixed the shape and the table's addresses, serves
//! every step of its rung with new table contents.
//!
//! Nothing here touches a device: the layouts, the padding and the refusals are tested
//! on the host.

use eidola_engine::spec::NULL_BLOCK;

use super::abi::DnSeq;
use crate::mimo::attention::{AttnRequest, HostPlan, PassCaps, Reduction, SPLIT_KEYS};
use crate::sampler::SampleRow;
use crate::{CudaError, Result, narrow};

/// The row-copy launches' buffer tables (`eidola_copy_rows` takes eight buffers of one
/// row width). A launch's items name these indices; the program binds each to its
/// buffer, per layer where the buffer is per layer.
pub mod buf {
    /// Rows of `conv_dim` f32: state images, live images and the tail at the layer, the
    /// staged conv rows, the widened new rows, three zero rows and the pad rows.
    pub const C_STATES: u32 = 0;
    pub const C_LIVE: u32 = 1;
    pub const C_TAIL: u32 = 2;
    pub const C_CONV: u32 = 3;
    pub const C_NEW: u32 = 4;
    pub const C_ZERO: u32 = 5;
    pub const C_PAD: u32 = 6;
    /// Rows of `2 · v_heads` f32 (`a` then `b`): the tail at the layer, the staged rows,
    /// the widened new rows, the pad rows.
    pub const A_TAIL: u32 = 0;
    pub const A_STAGED: u32 = 1;
    pub const A_NEW: u32 = 2;
    pub const A_PAD: u32 = 3;
    /// Rows of `v_heads · v_dim` f32: the recurrence's output by staged position, by
    /// token row, the pad rows.
    pub const O_STAGED: u32 = 0;
    pub const O_ROWS: u32 = 1;
    pub const O_PAD: u32 = 2;
    /// Rows of `hidden` BF16 (`hidden / 2` words): the slots' kept normed states, the
    /// drafter's taps, the target's normed rows, the drafter's normed rows, the launch's
    /// destination (the drafter's input, or the head's), the pad rows.
    pub const H_KEPT: u32 = 0;
    pub const H_TAPS: u32 = 1;
    pub const H_NORMED: u32 = 2;
    pub const H_DRAFT: u32 = 3;
    pub const H_DST: u32 = 4;
    pub const H_PAD: u32 = 5;
    /// One-word rows (token ids): the target's tokens (in the table: host ids, and the
    /// drafts once their copies have run), the draft lanes, the launch's destination token
    /// array, the pad rows.
    pub const T_HOST: u32 = 0;
    pub const T_LANES: u32 = 1;
    pub const T_DST: u32 = 2;
    pub const T_PAD: u32 = 3;
}

/// One row copy: row `src.1` of buffer `src.0` to row `dst.1` of buffer `dst.0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Item {
    pub src: (u32, u32),
    pub dst: (u32, u32),
}

impl Item {
    pub fn new(sb: u32, sr: u32, db: u32, dr: u32) -> Item {
        Item {
            src: (sb, sr),
            dst: (db, dr),
        }
    }

    /// The inert item `i` of a launch: pad row `i` onto itself, touching no row any other
    /// item names.
    pub fn pad(pad_buf: u32, i: u32) -> Item {
        Item::new(pad_buf, i, pad_buf, i)
    }
}

/// One attention launch's rows: per row its RoPE position, its query position and its KV
/// target (exact profile: also its block-table row); in the fast profile the FA2
/// requests over them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AttnRows {
    pub rope: Vec<u32>,
    pub query: Vec<u32>,
    pub targets: Vec<u32>,
    /// `[rows][table width]` block ids (exact profile).
    pub tables: Vec<u32>,
    /// FA2 requests covering the rows in order (fast profile).
    pub requests: Vec<AttnRequest>,
}

/// One forward through the target's layers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TargetPass {
    /// Token ids by row: each run's host ids, then a placeholder per draft.
    pub tokens: Vec<u32>,
    /// One-word copies of the drafts from their lanes over the placeholders.
    pub token_copies: Vec<Item>,
    /// The DeltaNet sequences, one a run.
    pub seqs: Vec<DnSeq>,
    /// Per staged position, the conv row of its oldest tap.
    pub window: Vec<u32>,
    /// Conv history and folded rows into the staged conv rows; the new rows after them.
    pub stage_conv: Vec<Item>,
    /// Folded and new `a`/`b` rows into the staged rows.
    pub stage_ab: Vec<Item>,
    /// After the recurrence: conv histories into stored images and live images, new
    /// positions into the tail.
    pub post_conv: Vec<Item>,
    pub post_ab: Vec<Item>,
    /// The recurrence's outputs of the new positions, by token row.
    pub o_rows: Vec<Item>,
    pub attn: AttnRows,
    /// Conv rows the staging uses.
    pub conv_rows: u32,
    /// Blocks of the blocked DeltaNet the runs take.
    pub tiles: u32,
}

impl TargetPass {
    /// Token rows.
    pub fn t(&self) -> usize {
        self.attn.rope.len()
    }
}

/// One forward of the drafter's layer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrafterPass {
    /// One-word copies into the drafter's token array.
    pub token_copies: Vec<Item>,
    /// Hidden rows into the drafter's input (`H_DST` is the input).
    pub hidden_copies: Vec<Item>,
    pub attn: AttnRows,
    /// Only the KV (the canonical rewrite): no attention, MLP or norm after the write.
    pub kv_only: bool,
    /// The rows whose logits are drawn, from the drafter's normed rows into the head's
    /// input (`H_DST` is the head's input).
    pub head: Vec<Item>,
    /// The draws from those logits (the draft stream), into the lanes of one depth.
    pub draws: Vec<SampleRow>,
}

/// A step's whole plan.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepPlan {
    /// The drafter at each drafting row's host position, then one pass a chain depth,
    /// each drawing a depth's drafts into the lanes: before the target, whose input the
    /// drafts are.
    pub lead: Vec<DrafterPass>,
    /// The target over every row's host tokens and, after them, its drafts.
    pub target: TargetPass,
    /// The target's normed rows into the head's input: every sampled row's last host
    /// row, then every draft's row.
    pub head: Vec<Item>,
    /// The drafter over the host positions of rows that do not draft, which may read
    /// the target's normed rows of this step.
    pub trail: Option<DrafterPass>,
    /// Rows without drafts, drawn from the target logits (tokens into the read-back).
    pub plain: Vec<SampleRow>,
    /// Drafting rows' target distributions (rows `p ..= p + k` of each, consecutive).
    pub targets: Vec<SampleRow>,
    /// Chain acceptance: per drafting row its sample row (position `p + 1`), its first
    /// target distribution and its draft count; its drafts and draft distributions are
    /// at index `d + i · D` of the lanes and draft distributions.
    pub accept: Vec<(SampleRow, u32, u32)>,
    /// The canonical rewrite of the drafter's KV at the drafts' positions.
    pub rewrite: Option<DrafterPass>,
    /// Normed rows into the slots' kept states and the drafter's taps.
    pub keep: Vec<Item>,
    /// Drafting rows (`D`): the stride of the draft lanes and distributions.
    pub drafting: u32,
    /// The widest draft count (`K`): the read-back's stride is `K + 1`.
    pub width: u32,
}

// ---- shapes ----

/// An FA2 launch's geometry: query heads per KV head, the page size, and the partial
/// rows the scratch holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fa2Geometry {
    pub group_size: u32,
    pub page_size: u32,
    pub partial_rows: u32,
}

/// What every launch of one attention takes as counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnShape {
    pub rows: usize,
    /// The exact kernel's block-table width (0 in the fast profile).
    pub table_width: usize,
    /// The FA2 pass's reservation (fast profile).
    pub fa2: Option<PassCaps>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetShape {
    pub token_copies: usize,
    pub runs: usize,
    pub staged: usize,
    pub conv_rows: usize,
    /// Blocks of the blocked DeltaNet the launch's scratch is numbered for.
    pub tiles: usize,
    pub stage_conv: usize,
    pub stage_ab: usize,
    pub post_conv: usize,
    pub post_ab: usize,
    pub attn: AttnShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrafterShape {
    pub kv_only: bool,
    pub attn: AttnShape,
    pub head: usize,
    pub draws: usize,
}

/// Every count of a step's launches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub lead: Vec<DrafterShape>,
    pub target: TargetShape,
    pub head: usize,
    pub trail: Option<DrafterShape>,
    pub plain: usize,
    pub targets: usize,
    pub accept: usize,
    pub rewrite: Option<DrafterShape>,
    pub keep: usize,
    pub drafting: u32,
    pub width: u32,
}

/// What the packer needs of the executor: the profile's attention and its geometries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geo {
    pub fast: bool,
    pub table_width: usize,
    pub target: Fa2Geometry,
    pub drafter: Fa2Geometry,
}

/// The FA2 plan of `requests`, one pass.
pub fn fa2_plan(requests: &[AttnRequest], g: &Fa2Geometry) -> Result<HostPlan> {
    let plan = HostPlan::with_max_tile(
        requests,
        g.group_size,
        g.page_size,
        Reduction::Split,
        g.partial_rows,
        super::attention::MAX_TILE,
    )?;
    if plan.passes().len() != 1 {
        return Err(CudaError::new(format!(
            "attention: {} passes; the partial rows hold one",
            plan.passes().len()
        )));
    }
    Ok(plan)
}

/// An eager launch's reservation: exactly its plan's own.
fn own_caps(plan: &HostPlan) -> Result<PassCaps> {
    let p = &plan.passes()[0];
    let shape = p.shape(plan)?;
    Ok(PassCaps {
        shape,
        work_items: shape.grid,
        pages: narrow(p.indices.len(), "pages")?,
    })
}

fn attn_shape(a: &AttnRows, geo: &Geo, g: &Fa2Geometry) -> Result<AttnShape> {
    Ok(AttnShape {
        rows: a.rope.len(),
        table_width: if geo.fast { 0 } else { geo.table_width },
        fa2: if geo.fast && !a.requests.is_empty() {
            Some(own_caps(&fa2_plan(&a.requests, g)?)?)
        } else {
            None
        },
    })
}

impl Shape {
    /// A plan's own shape: the eager step's.
    pub fn of(p: &StepPlan, geo: &Geo) -> Result<Shape> {
        let target = |t: &TargetPass| -> Result<TargetShape> {
            Ok(TargetShape {
                token_copies: t.token_copies.len(),
                runs: t.seqs.len(),
                staged: t.window.len(),
                conv_rows: t.conv_rows as usize,
                tiles: t.tiles as usize,
                stage_conv: t.stage_conv.len(),
                stage_ab: t.stage_ab.len(),
                post_conv: t.post_conv.len(),
                post_ab: t.post_ab.len(),
                attn: attn_shape(&t.attn, geo, &geo.target)?,
            })
        };
        let drafter = |d: &DrafterPass| -> Result<DrafterShape> {
            let mut attn = attn_shape(&d.attn, geo, &geo.drafter)?;
            if d.kv_only {
                // Only the KV write: no attention reads.
                attn.fa2 = None;
                attn.table_width = 0;
            }
            Ok(DrafterShape {
                kv_only: d.kv_only,
                attn,
                head: d.head.len(),
                draws: d.draws.len(),
            })
        };
        Ok(Shape {
            lead: p.lead.iter().map(drafter).collect::<Result<_>>()?,
            target: target(&p.target)?,
            head: p.head.len(),
            trail: p.trail.as_ref().map(drafter).transpose()?,
            plain: p.plain.len(),
            targets: p.targets.len(),
            accept: p.accept.len(),
            rewrite: p.rewrite.as_ref().map(drafter).transpose()?,
            keep: p.keep.len(),
            drafting: p.drafting,
            width: p.width,
        })
    }

    /// Target logit rows: the head's.
    pub fn target_logits(&self) -> usize {
        self.head
    }
}

// ---- the table's layout ----

/// An array of the table: its first word and its length in words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Arr {
    pub at: usize,
    pub len: usize,
}

/// A copy launch's four item arrays (`items` words each).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ItemsAt {
    pub src_buf: Arr,
    pub src_row: Arr,
    pub dst_buf: Arr,
    pub dst_row: Arr,
    pub items: usize,
}

/// One FA2 pass's arrays (a `PaddedPass`'s), and its reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fa2At {
    pub caps: PassCaps,
    pub q_indptr: Arr,
    pub indices: Arr,
    pub indptr: Arr,
    pub last_page_len: Arr,
    pub request_indices: Arr,
    pub qo_tile_indices: Arr,
    pub kv_tile_indices: Arr,
    pub o_indptr: Arr,
    pub merge_indptr: Arr,
    /// The work-item count, then the KV chunk size.
    pub counts: Arr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnAt {
    pub rows: usize,
    pub rope: Arr,
    pub query: Arr,
    pub targets: Arr,
    pub tables: Arr,
    pub table_width: usize,
    pub fa2: Option<Fa2At>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetAt {
    pub tokens: Arr,
    pub token_copies: ItemsAt,
    pub seqs: Arr,
    pub runs: usize,
    pub tiles: usize,
    pub window: Arr,
    pub conv_rows: usize,
    pub stage_conv: ItemsAt,
    pub stage_ab: ItemsAt,
    pub post_conv: ItemsAt,
    pub post_ab: ItemsAt,
    pub o_rows: ItemsAt,
    pub attn: AttnAt,
}

impl TargetAt {
    pub fn t(&self) -> usize {
        self.attn.rows
    }
    pub fn staged(&self) -> usize {
        self.window.len
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrafterAt {
    pub token_copies: ItemsAt,
    pub hidden_copies: ItemsAt,
    pub kv_only: bool,
    pub attn: AttnAt,
    pub head: ItemsAt,
    pub draws: Arr,
}

/// Words of one [`SampleRow`].
pub const SAMPLE_WORDS: usize = 8;
/// Words of one [`DnSeq`].
pub const SEQ_WORDS: usize = std::mem::size_of::<DnSeq>() / 4;

/// Every array of a step's table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub lead: Vec<DrafterAt>,
    pub target: TargetAt,
    pub head: ItemsAt,
    pub trail: Option<DrafterAt>,
    pub plain: Arr,
    pub targets: Arr,
    /// Acceptance: sample rows, target rows, draft rows (`d`), draft counts.
    pub accept_rows: Arr,
    pub accept_target: Arr,
    pub accept_draft: Arr,
    pub accept_drafts: Arr,
    pub accepts: usize,
    pub rewrite: Option<DrafterAt>,
    pub keep: ItemsAt,
    pub drafting: u32,
    pub width: u32,
    /// Words of the table.
    pub words: usize,
}

struct Arena(usize);

impl Arena {
    /// An array of `len` words at an even word (8-byte aligned).
    fn arr(&mut self, len: usize) -> Arr {
        let at = self.0.next_multiple_of(2);
        self.0 = at + len;
        Arr { at, len }
    }

    fn items(&mut self, items: usize) -> ItemsAt {
        ItemsAt {
            src_buf: self.arr(items),
            src_row: self.arr(items),
            dst_buf: self.arr(items),
            dst_row: self.arr(items),
            items,
        }
    }

    fn attn(&mut self, s: &AttnShape) -> AttnAt {
        AttnAt {
            rows: s.rows,
            rope: self.arr(s.rows),
            query: self.arr(s.rows),
            targets: self.arr(s.rows),
            tables: self.arr(s.rows * s.table_width),
            table_width: s.table_width,
            fa2: s.fa2.map(|caps| Fa2At {
                caps,
                q_indptr: self.arr(caps.request_words()),
                indptr: self.arr(caps.request_words()),
                last_page_len: self.arr(caps.request_words() - 1),
                o_indptr: self.arr(caps.request_words()),
                merge_indptr: self.arr(caps.merge_words()),
                request_indices: self.arr(caps.work_items as usize),
                qo_tile_indices: self.arr(caps.work_items as usize),
                kv_tile_indices: self.arr(caps.work_items as usize),
                counts: self.arr(2),
                indices: self.arr(caps.pages as usize),
            }),
        }
    }

    fn target(&mut self, s: &TargetShape) -> TargetAt {
        let t = s.attn.rows;
        TargetAt {
            tokens: self.arr(t),
            token_copies: self.items(s.token_copies),
            seqs: self.arr(s.runs * SEQ_WORDS),
            runs: s.runs,
            tiles: s.tiles,
            window: self.arr(s.staged),
            conv_rows: s.conv_rows,
            stage_conv: self.items(s.stage_conv),
            stage_ab: self.items(s.stage_ab),
            post_conv: self.items(s.post_conv),
            post_ab: self.items(s.post_ab),
            o_rows: self.items(t),
            attn: self.attn(&s.attn),
        }
    }

    fn drafter(&mut self, s: &DrafterShape, draws: usize) -> DrafterAt {
        let n = s.attn.rows;
        DrafterAt {
            token_copies: self.items(n),
            hidden_copies: self.items(n),
            kv_only: s.kv_only,
            attn: self.attn(&s.attn),
            head: self.items(s.head),
            draws: self.arr(draws * SAMPLE_WORDS),
        }
    }
}

impl Layout {
    /// Every array at an offset that is a function of `s` alone.
    pub fn new(s: &Shape) -> Layout {
        let mut m = Arena(0);
        let lead = s.lead.iter().map(|d| m.drafter(d, d.draws)).collect();
        let target = m.target(&s.target);
        let head = m.items(s.head);
        let trail = s.trail.as_ref().map(|d| m.drafter(d, d.draws));
        let plain = m.arr(s.plain * SAMPLE_WORDS);
        let targets = m.arr(s.targets * SAMPLE_WORDS);
        let accept_rows = m.arr(s.accept * SAMPLE_WORDS);
        let accept_target = m.arr(s.accept);
        let accept_draft = m.arr(s.accept);
        let accept_drafts = m.arr(s.accept);
        let rewrite = s.rewrite.as_ref().map(|r| m.drafter(r, 0));
        let keep = m.items(s.keep);
        Layout {
            lead,
            target,
            head,
            trail,
            plain,
            targets,
            accept_rows,
            accept_target,
            accept_draft,
            accept_drafts,
            accepts: s.accept,
            rewrite,
            keep,
            drafting: s.drafting,
            width: s.width,
            words: m.0.next_multiple_of(64),
        }
    }

    /// Words of the read-back: the status word, the plain rows' tokens, per drafting row
    /// `K + 1` accepted tokens, then their counts.
    pub fn readback_words(&self) -> usize {
        1 + self.plain.len / SAMPLE_WORDS + self.accepts * (self.width as usize + 1) + self.accepts
    }
}

// ---- packing ----

fn refuse(what: String) -> CudaError {
    CudaError::new(format!("step table: {what}"))
}

fn put(words: &mut [u32], a: Arr, v: &[u32], what: &str) -> Result<()> {
    if v.len() > a.len {
        return Err(refuse(format!("{what}: {} words past {}", v.len(), a.len)));
    }
    words[a.at..a.at + v.len()].copy_from_slice(v);
    Ok(())
}

fn exact(words: &mut [u32], a: Arr, v: &[u32], what: &str) -> Result<()> {
    if v.len() != a.len {
        return Err(refuse(format!(
            "{what}: {} words where the shape has {}",
            v.len(),
            a.len
        )));
    }
    put(words, a, v, what)
}

/// Items, padded with inert copies of `pad` rows to the launch's count.
fn items(words: &mut [u32], at: &ItemsAt, v: &[Item], pad: u32, what: &str) -> Result<()> {
    if v.len() > at.items {
        return Err(refuse(format!(
            "{what}: {} items past {}",
            v.len(),
            at.items
        )));
    }
    for i in 0..at.items {
        let it = v
            .get(i)
            .copied()
            .unwrap_or_else(|| Item::pad(pad, u32::try_from(i).expect("a launch's items")));
        words[at.src_buf.at + i] = it.src.0;
        words[at.src_row.at + i] = it.src.1;
        words[at.dst_buf.at + i] = it.dst.0;
        words[at.dst_row.at + i] = it.dst.1;
    }
    Ok(())
}

fn words_of<T: Copy>(v: &[T]) -> Vec<u32> {
    let bytes = std::mem::size_of_val(v);
    debug_assert!(bytes.is_multiple_of(4));
    // SAFETY: `repr(C)` plain data of whole words.
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), bytes) };
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn attn(words: &mut [u32], at: &AttnAt, a: &AttnRows, g: &Fa2Geometry, what: &str) -> Result<()> {
    exact(words, at.rope, &a.rope, what)?;
    exact(words, at.query, &a.query, what)?;
    exact(words, at.targets, &a.targets, what)?;
    if at.table_width > 0 {
        exact(words, at.tables, &a.tables, what)?;
    }
    if let Some(f) = &at.fa2 {
        let p = fa2_plan(&a.requests, g)?.padded(&f.caps)?;
        exact(words, f.q_indptr, &p.q_indptr, what)?;
        exact(words, f.indptr, &p.indptr, what)?;
        exact(words, f.last_page_len, &p.last_page_len, what)?;
        exact(words, f.o_indptr, &p.o_indptr, what)?;
        exact(words, f.merge_indptr, &p.merge_indptr, what)?;
        put(words, f.request_indices, &p.request_indices, what)?;
        put(words, f.qo_tile_indices, &p.qo_tile_indices, what)?;
        put(words, f.kv_tile_indices, &p.kv_tile_indices, what)?;
        put(words, f.indices, &p.indices, what)?;
        put(words, f.counts, &[p.work_items, SPLIT_KEYS], what)?;
    }
    Ok(())
}

fn target(words: &mut [u32], at: &TargetAt, t: &TargetPass, geo: &Geo, what: &str) -> Result<()> {
    exact(words, at.tokens, &t.tokens, what)?;
    if t.token_copies.len() != at.token_copies.items {
        return Err(refuse(format!("{what}: token copies")));
    }
    items(words, &at.token_copies, &t.token_copies, buf::T_PAD, what)?;
    exact(words, at.seqs, &words_of(&t.seqs), what)?;
    if t.window.len() > at.window.len || t.conv_rows as usize > at.conv_rows {
        return Err(refuse(format!("{what}: staged positions past the shape")));
    }
    if t.tiles as usize > at.tiles {
        return Err(refuse(format!("{what}: blocks past the shape")));
    }
    // Positions past the plan's stage from conv row 0: computed, never read.
    let mut window = t.window.clone();
    window.resize(at.window.len, 0);
    exact(words, at.window, &window, what)?;
    items(words, &at.stage_conv, &t.stage_conv, buf::C_PAD, what)?;
    items(words, &at.stage_ab, &t.stage_ab, buf::A_PAD, what)?;
    items(words, &at.post_conv, &t.post_conv, buf::C_PAD, what)?;
    items(words, &at.post_ab, &t.post_ab, buf::A_PAD, what)?;
    if t.o_rows.len() != at.o_rows.items {
        return Err(refuse(format!("{what}: output rows")));
    }
    items(words, &at.o_rows, &t.o_rows, buf::O_PAD, what)?;
    attn(words, &at.attn, &t.attn, &geo.target, what)
}

fn drafter(
    words: &mut [u32],
    at: &DrafterAt,
    d: &DrafterPass,
    geo: &Geo,
    what: &str,
) -> Result<()> {
    if d.token_copies.len() != at.token_copies.items
        || d.hidden_copies.len() != at.hidden_copies.items
        || d.head.len() != at.head.items
        || d.kv_only != at.kv_only
    {
        return Err(refuse(format!("{what}: rows")));
    }
    items(words, &at.token_copies, &d.token_copies, buf::T_PAD, what)?;
    items(words, &at.hidden_copies, &d.hidden_copies, buf::H_PAD, what)?;
    items(words, &at.head, &d.head, buf::H_PAD, what)?;
    exact(words, at.draws, &words_of(&d.draws), what)?;
    attn(words, &at.attn, &d.attn, &geo.drafter, what)
}

/// The table of `p` in `layout`: refused unless every fixed count is the plan's and every
/// variable list fits.
pub fn pack(p: &StepPlan, layout: &Layout, geo: &Geo) -> Result<Vec<u32>> {
    let mut w = vec![0u32; layout.words];
    let words = &mut w;
    if p.lead.len() != layout.lead.len()
        || p.trail.is_some() != layout.trail.is_some()
        || p.rewrite.is_some() != layout.rewrite.is_some()
        || p.drafting != layout.drafting
        || p.width != layout.width
    {
        return Err(refuse("the plan's passes are not the shape's".into()));
    }
    for (d, at) in p.lead.iter().zip(&layout.lead) {
        drafter(words, at, d, geo, "drafter")?;
    }
    target(words, &layout.target, &p.target, geo, "target")?;
    if p.head.len() != layout.head.items {
        return Err(refuse("head rows".into()));
    }
    items(words, &layout.head, &p.head, buf::H_PAD, "head")?;
    if let (Some(d), Some(at)) = (&p.trail, &layout.trail) {
        drafter(words, at, d, geo, "trailing drafter")?;
    }
    exact(words, layout.plain, &words_of(&p.plain), "plain rows")?;
    exact(words, layout.targets, &words_of(&p.targets), "target rows")?;
    let rows: Vec<SampleRow> = p.accept.iter().map(|a| a.0).collect();
    exact(words, layout.accept_rows, &words_of(&rows), "accept rows")?;
    let col = |f: fn(&(SampleRow, u32, u32)) -> u32| p.accept.iter().map(f).collect::<Vec<_>>();
    exact(words, layout.accept_target, &col(|a| a.1), "accept")?;
    let d: Vec<u32> = (0..p.accept.len())
        .map(|i| u32::try_from(i).expect("a row"))
        .collect();
    exact(words, layout.accept_draft, &d, "accept")?;
    exact(words, layout.accept_drafts, &col(|a| a.2), "accept")?;
    if let (Some(r), Some(at)) = (&p.rewrite, &layout.rewrite) {
        drafter(words, at, r, geo, "rewrite")?;
    }
    if p.keep.len() > layout.keep.items {
        return Err(refuse("kept rows".into()));
    }
    items(words, &layout.keep, &p.keep, buf::H_PAD, "kept rows")?;
    Ok(w)
}

/// A request over `rows` consecutive queries ending at key index `last` (relative
/// positions), its pages `page(j)` for every page index of keys `0 ..= last`.
pub fn request(
    q_start: u32,
    rows: u32,
    last: u32,
    page_size: u32,
    page: &mut dyn FnMut(u32) -> u32,
) -> AttnRequest {
    let kv_len = last + 1;
    AttnRequest {
        q_start,
        qo_len: rows,
        pages: (0..kv_len.div_ceil(page_size)).map(page).collect(),
        kv_start: 0,
        kv_len,
    }
}

/// The page every key of a padding request reads: the pad block (target) or its first
/// row (drafter), which no table maps.
pub fn pad_page(pad_block: u32, page_size: u32, block_size: u32) -> u32 {
    if page_size == 1 {
        u32::try_from(super::attention::drafter_page(pad_block, 0, block_size))
            .expect("a page id within the pool")
    } else {
        pad_block
    }
}

/// `NULL_BLOCK` as a table word.
pub const NULL: u32 = NULL_BLOCK;

/// The partial rows FA2's scratch needs for `rows` query rows reaching `positions`.
pub fn partial_rows(rows: u32, positions: u32) -> u32 {
    rows.saturating_mul(positions.div_ceil(SPLIT_KEYS).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo(fast: bool) -> Geo {
        let g = |page_size| Fa2Geometry {
            group_size: 6,
            page_size,
            partial_rows: 4096,
        };
        Geo {
            fast,
            table_width: 8,
            target: g(4),
            drafter: g(1),
        }
    }

    fn row_pass(rows: u32, staged: u32, fast: bool) -> TargetPass {
        let mut t = TargetPass::default();
        for r in 0..rows {
            t.tokens.push(r + 7);
            t.seqs.push(DnSeq {
                first: r,
                count: 1,
                ..DnSeq::default()
            });
            t.attn.rope.push(9 + r);
            t.attn.query.push(9 + r);
            t.attn.targets.push(40 + r);
            if !fast {
                t.attn.tables.extend([1u32; 8]);
            }
            t.attn
                .requests
                .push(request(r, 1, 9 + r, 4, &mut |j| j + 1));
            t.o_rows.push(Item::new(buf::O_STAGED, r, buf::O_ROWS, r));
        }
        t.window = (0..staged).collect();
        t.conv_rows = 3 * rows + staged;
        t.stage_conv = (0..staged)
            .map(|i| Item::new(buf::C_NEW, i, buf::C_CONV, i))
            .collect();
        t
    }

    #[test]
    fn a_shape_lays_out_the_same_offsets_whatever_the_contents() {
        for fast in [false, true] {
            let g = geo(fast);
            let mut p = StepPlan {
                target: row_pass(3, 5, fast),
                ..StepPlan::default()
            };
            let s = Shape::of(&p, &g).expect("shape");
            let l = Layout::new(&s);
            let one = pack(&p, &l, &g).expect("pack");
            p.target.tokens = vec![1, 2, 3];
            p.target.stage_conv.truncate(2);
            p.target.window.truncate(4);
            let two = pack(&p, &l, &g).expect("pack");
            assert_eq!(Layout::new(&s), l);
            assert_eq!(one.len(), two.len());
            // The truncated lists are padded: inert self-copies, window rows at 0.
            let at = l.target.stage_conv;
            assert_eq!(two[at.src_buf.at + 3], buf::C_PAD);
            assert_eq!(two[at.src_row.at + 3], 3);
            assert_eq!(two[at.dst_row.at + 3], 3);
            assert_eq!(two[l.target.window.at + 4], 0);
            // Every array of an even word, inside the table.
            for a in [
                l.target.seqs,
                l.target.window,
                l.target.tokens,
                l.target.attn.rope,
                l.plain,
            ] {
                assert!(a.at.is_multiple_of(2) && a.at + a.len <= l.words);
            }
        }
    }

    #[test]
    fn a_plan_past_its_shape_is_refused() {
        let g = geo(false);
        let p = StepPlan {
            target: row_pass(3, 5, false),
            ..StepPlan::default()
        };
        let l = Layout::new(&Shape::of(&p, &g).expect("shape"));
        let mut more = p.clone();
        more.target.stage_conv.push(Item::default());
        assert!(pack(&more, &l, &g).is_err());
        let mut longer = p.clone();
        longer.target.window.push(0);
        assert!(pack(&longer, &l, &g).is_err());
        let mut fewer = p.clone();
        fewer.target.tokens.pop();
        assert!(pack(&fewer, &l, &g).is_err());
        let mut drafting = p;
        drafting.lead.push(DrafterPass::default());
        assert!(pack(&drafting, &l, &g).is_err());
    }

    #[test]
    fn drafts_join_the_host_tokens_in_one_target_pass() {
        // Two drafting rows of three drafts each: one run each of four positions, the
        // host id first, the drafts copied from their lanes over the placeholders.
        let g = geo(false);
        let mut p = StepPlan {
            target: row_pass(8, 8, false),
            width: 3,
            drafting: 2,
            ..StepPlan::default()
        };
        p.target.tokens = vec![7, 0, 0, 0, 9, 0, 0, 0];
        for (d, first) in [(0u32, 0u32), (1, 4)] {
            for j in 0..3 {
                p.target.token_copies.push(Item::new(
                    buf::T_LANES,
                    j * 2 + d,
                    buf::T_DST,
                    first + 1 + j,
                ));
            }
        }
        let l = Layout::new(&Shape::of(&p, &g).expect("shape"));
        let w = pack(&p, &l, &g).expect("pack");
        let at = l.target.tokens;
        assert_eq!(&w[at.at..at.at + 8], &[7, 0, 0, 0, 9, 0, 0, 0]);
        let c = l.target.token_copies;
        assert_eq!(c.items, 6);
        assert_eq!(
            w[c.src_row.at + 4],
            3,
            "the second row's second draft's lane"
        );
        assert_eq!(w[c.dst_row.at + 4], 6);
        // A draft without its copy is refused.
        let mut short = p;
        short.target.token_copies.pop();
        assert!(pack(&short, &l, &g).is_err());
    }

    #[test]
    fn fa2_requests_pack_into_a_rungs_reservation() {
        let g = geo(true);
        // A rung of 4 decode rows reaching key 2,100 (three chunks of 1,024).
        let caps = PassCaps::with_max_tile(
            Reduction::Split,
            4,
            1,
            6,
            4,
            600,
            2200,
            super::super::attention::MAX_TILE,
        )
        .expect("caps");
        let mut p = StepPlan {
            target: row_pass(4, 4, true),
            ..StepPlan::default()
        };
        p.target.attn.requests = (0..4)
            .map(|r| request(r, 1, 100 + 600 * r, 4, &mut |j| j + 1))
            .collect();
        let mut s = Shape::of(&p, &g).expect("shape");
        s.target.attn.fa2 = Some(caps);
        let l = Layout::new(&s);
        let w = pack(&p, &l, &g).expect("pack");
        let f = l.target.attn.fa2.expect("fa2");
        // Work items: one per row and chunk (1 + 1 + 2 + 2), walked by the fixed grid.
        assert_eq!(w[f.counts.at], 6);
        assert_eq!(w[f.counts.at + 1], SPLIT_KEYS);
        assert_eq!(caps.shape.grid, caps.work_items.min(148));
        // Rows past the reservation's pages (600 a row) are refused.
        p.target.attn.requests = (0..4)
            .map(|r| request(r, 1, 2403, 4, &mut |j| j + 1))
            .collect();
        assert!(pack(&p, &l, &g).is_err());
    }

    #[test]
    fn drafter_requests_count_keys_from_position_one() {
        // A drafter row at s = 9 sees positions 1 ..= 9: nine relative keys, one page each.
        let mut pages = Vec::new();
        let r = request(0, 1, 8, 1, &mut |j| {
            pages.push(j + 1);
            j + 1
        });
        assert_eq!(r.kv_len, 9);
        assert_eq!(r.pages.len(), 9);
        assert_eq!(pages.first(), Some(&1));
        assert_eq!(pages.last(), Some(&9));
        assert_eq!(pad_page(5, 1, 4), 40);
        assert_eq!(pad_page(5, 4, 4), 5);
    }
}
