//! Both halves of the released DeepSeek-V4.1 model in one process, shared by the
//! dev-only examples that run them together (`ds_whole`: one GPU, the routed experts
//! staged; `ds_two`: two GPUs, every weight resident).
//!
//! A [`Layout`] is the halves' configuration for a few token sequences, each in its own
//! state slot with its own blocks, every block of it mapped. A [`Seq`] runs one sequence
//! through the executor's own step path in chunks (`DsCudaExecutor::encode_half`, the
//! Engram rows the decoder's device holds gathered there, the crossing,
//! `DsCudaExecutor::decode_half` with logits at every position) and leaves a
//! [`Recording`] in the form `ds_halves` writes, which [`Recording::compare`] holds to
//! that example's recordings bit for bit, naming the first difference by array, position
//! and element.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use eidola_engine::executor::Slot;
use eidola_engine::spec::{Bucket, ModelSpec};
use eidola_engine_cuda::ds::abi::EngramGatherParams;
use eidola_engine_cuda::ds::boundary::{GatheredRows, ToDecoder, ToEncoder};
use eidola_engine_cuda::ds::executor::geometry;
use eidola_engine_cuda::ds::gemm::GemmForm;
use eidola_engine_cuda::ds::half::{Device, bytes_of};
use eidola_engine_cuda::ds::kv::KvStore;
use eidola_engine_cuda::ds::model::{HalfNet, MemoryBytes};
use eidola_engine_cuda::ds::plan::{PlanRow, StepPlan};
use eidola_engine_cuda::ds::support::DsShape;
use eidola_engine_cuda::ds::weights::{Half, HalfWeights};
use eidola_engine_cuda::ds::{DsCudaExecutor, DsCudaExecutorConfig};
use eidola_engine_cuda::{CudaError, Result};
use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::DsConfig;
use eidola_engine_model::deepseek::engram::EngramHasher;
use serde_json::{Value, json};

pub fn fail(e: impl std::fmt::Display) -> CudaError {
    CudaError::new(e.to_string())
}

pub fn us(x: u32) -> usize {
    x as usize
}

pub fn n32(x: usize) -> u32 {
    u32::try_from(x).expect("a count of this run")
}

/// Positions per KV block, as `ds_halves`.
pub const BLOCK: u32 = 16;

/// The halves' configuration for sequences of at most `positions` positions, one state
/// slot each, slot `s` holding blocks `1 + s · per_slot ..`.
pub struct Layout {
    /// The checkpoint; none lays the halves out without it (a dry run).
    pub store: Option<Arc<WeightSet>>,
    pub c: DsConfig,
    pub shape: DsShape,
    pub spec: ModelSpec,
    pub chunk: u32,
    pub slots: u32,
    pub hasher: Option<EngramHasher>,
}

impl Layout {
    pub fn new(
        store: Option<Arc<WeightSet>>,
        c: DsConfig,
        token_map: Vec<u32>,
        longest: usize,
        slots: u32,
        chunk: u32,
    ) -> Result<Layout> {
        let positions = n32(longest).next_multiple_of(BLOCK);
        let cfg = DsCudaExecutorConfig {
            block_size: BLOCK,
            num_blocks: slots * (positions / BLOCK) + 2,
            num_state_slots: slots,
            max_model_len: positions,
            buckets: vec![Bucket {
                max_seqs: chunk,
                max_tokens: chunk,
            }],
            sampleable_vocab_size: n32(c.vocab_size),
            devices: [0, 0],
            draft_width: 0,
            host_blocks: 0,
            position_tags: false,
            record: false,
            gemm: GemmForm::from_env()?,
        };
        let (shape, spec) = DsCudaExecutor::preflight(&c, &cfg)?;
        let hasher = match &c.engram {
            Some(e) => Some(EngramHasher::new(e, c.vocab_size, token_map).map_err(fail)?),
            None => None,
        };
        Ok(Layout {
            store,
            c,
            shape,
            spec,
            chunk,
            slots,
            hasher,
        })
    }

    /// Elements of a position's residual copies.
    pub fn width(&self) -> usize {
        self.c.hc_mult * self.c.dim
    }

    pub fn entry_bytes(&self) -> usize {
        let (w, i) = (self.c.attention.head_dim, self.c.indexer.head_dim);
        w / 2 + w / 16 + i / 2 + i / 32
    }

    /// Every block of every slot's sequence, mapped.
    fn per_slot(&self) -> u32 {
        self.spec.max_blocks_per_seq()
    }

    /// Every half's weights as the executor loads them, with the Engram tables its
    /// device holds.
    pub fn resident(&self, half: Half, dev: &dyn Device) -> Result<HalfWeights> {
        HalfWeights::load(
            dev,
            self.store.as_deref(),
            &self.c,
            &self.shape,
            half,
            self.spec.max_model_len,
            true,
        )
    }

    /// A half over `w` on `dev`, with scratch for a chunk and every slot's blocks mapped.
    pub fn net(&self, half: Half, dev: Box<dyn Device>, w: HalfWeights) -> Result<HalfNet> {
        let positions = self.spec.max_model_len;
        let kv = KvStore::new(
            dev.as_ref(),
            geometry(&self.c, &self.spec, half)?,
            self.slots,
            self.per_slot(),
        )?;
        let mut n = HalfNet::new(
            half,
            dev,
            w,
            kv,
            &self.c,
            &self.shape,
            positions,
            self.chunk,
            self.chunk,
            self.chunk,
            self.spec.sampleable_vocab_size,
            self.shape.open_cap(),
        )?;
        let per = self.per_slot();
        for g in 0..n.kv.groups.len() {
            for slot in 0..self.slots {
                for index in 0..per {
                    n.kv.set_table(g, slot, index, 1 + slot * per + index);
                }
            }
        }
        n.kv.flush_tables(n.dev.as_ref())?;
        Ok(n)
    }

    pub fn seq<'a>(&'a self, tokens: &'a [u32], slot: Slot) -> Seq<'a> {
        assert!(slot < self.slots, "slot {slot} of {}", self.slots);
        assert!(tokens.len() <= us(self.spec.max_model_len));
        Seq {
            l: self,
            tokens,
            slot,
        }
    }
}

/// One sequence of a [`Layout`] in its slot.
pub struct Seq<'a> {
    pub l: &'a Layout,
    pub tokens: &'a [u32],
    pub slot: Slot,
}

/// What one step of both halves left.
pub struct Step {
    pub items: Vec<(Slot, usize)>,
    pub encoder: Vec<Vec<u16>>,
    pub crossing: ToDecoder,
    pub decoder: Vec<Vec<u16>>,
    pub logits: Vec<f32>,
    /// Gather, encoder half, decoder half, whole step.
    pub seconds: [f64; 4],
}

/// Watches the halves around each step (a staged half's caches, say).
pub trait Watch {
    type Snap;
    fn snap(&self, enc: &HalfNet, dec: &HalfNet) -> Self::Snap;
    /// What the step between two snapshots did, recorded with the step.
    fn delta(&self, before: &Self::Snap, after: &Self::Snap) -> Value;
}

/// Watches nothing.
pub struct Unwatched;

impl Watch for Unwatched {
    type Snap = ();
    fn snap(&self, _: &HalfNet, _: &HalfNet) {}
    fn delta(&self, _: &(), _: &()) -> Value {
        Value::Null
    }
}

/// A sequence's run through both halves, in `ds_halves`' recording form.
pub struct Recording {
    /// Per encoder layer, the residual copies leaving it at every position.
    pub encoder: Vec<Vec<u16>>,
    pub entries: Vec<u8>,
    pub streams: Vec<u16>,
    pub pre: Vec<f32>,
    pub decoder: Vec<Vec<u16>>,
    /// `[positions, vocab]`.
    pub logits: Vec<f32>,
    pub steps: Vec<Value>,
}

impl Seq<'_> {
    pub fn chunks(&self) -> Vec<(usize, usize)> {
        (0..self.tokens.len())
            .step_by(us(self.l.chunk))
            .map(|from| (from, (from + us(self.l.chunk)).min(self.tokens.len())))
            .collect()
    }

    fn plan(&self, from: usize, to: usize) -> (Vec<PlanRow>, StepPlan) {
        let rows = vec![PlanRow {
            slot: self.slot,
            c: n32(from),
            p: n32(to - 1),
            sample: false,
            chain_start: n32(from),
            drafts: 0,
        }];
        let chain = self.l.spec.window_chain.expect("a window chain");
        let plan = StepPlan::new(&rows, chain.depth, chain.window);
        (rows, plan)
    }

    /// Per Engram layer of the config, the table rows of `items`' positions.
    fn engram_ids(&self, items: &[(Slot, usize)]) -> Result<Vec<Vec<u32>>> {
        let (Some(e), Some(hasher)) = (&self.l.c.engram, &self.l.hasher) else {
            return Ok(Vec::new());
        };
        let mut out = vec![Vec::new(); e.layers.len()];
        for (slot, spec) in e.layers.iter().enumerate() {
            if self.l.shape.engram_layer[slot].is_none() {
                continue;
            }
            for &(_, pos) in items {
                let lookback = &self.tokens[(pos + 1).saturating_sub(e.max_ngram)..=pos];
                for row in hasher.rows(spec, lookback, pos) {
                    out[slot].push(u32::try_from(row).map_err(fail)?);
                }
            }
        }
        Ok(out)
    }

    /// The decoder half's part of the Engram lookups, gathered from the tables on its
    /// device (as the executor's step gathers them).
    fn gather(&self, n: &HalfNet, ids: &[Vec<u32>], rows: u32) -> Result<ToEncoder> {
        let dev = n.dev.as_ref();
        let mut gathered = vec![None; ids.len()];
        for (slot, ids) in ids.iter().enumerate() {
            let Some(table) = n.w.tables.get(slot).copied().flatten() else {
                continue;
            };
            n.s.eids.write(dev, 0, ids)?;
            let (codes, scales) = &n.s.gathered[slot];
            // SAFETY: the table, and `rows` rows of ids and of gathered rows.
            unsafe {
                dev.kernels().engram_gather(&EngramGatherParams {
                    codes: table.codes,
                    scales: table.scales,
                    ids: n.s.eids.addr,
                    out_codes: codes.addr,
                    out_scales: scales.addr,
                    status: n.s.status.addr,
                    table_rows: table.rows,
                    rows,
                    ids_per_row: n.d.lookups,
                    width: n.d.engram_width / n.d.lookups,
                    reserved: 0,
                })?;
            }
            let width = us(n.d.engram_width);
            gathered[slot] = Some(GatheredRows {
                codes: codes.read(dev, 0, us(rows) * width)?,
                scales: scales.read(dev, 0, us(rows) * width / 32)?,
            });
        }
        Ok(ToEncoder { gathered })
    }

    /// Positions `from .. to` through both halves.
    pub fn step(
        &self,
        enc: &mut HalfNet,
        dec: &mut HalfNet,
        from: usize,
        to: usize,
    ) -> Result<Step> {
        let (c, shape) = (&self.l.c, &self.l.shape);
        let began = Instant::now();
        let (plan_rows, plan) = self.plan(from, to);
        let items: Vec<(Slot, usize)> = plan.items.iter().map(|it| (it.slot, us(it.pos))).collect();
        let tokens: Vec<u32> = items.iter().map(|(_, p)| self.tokens[*p]).collect();
        let (count, width) = (items.len(), self.l.width());
        let ids = self.engram_ids(&items)?;
        let to_encoder = self.gather(dec, &ids, n32(count))?;
        let gather = began.elapsed().as_secs_f64();
        let at = Instant::now();
        let mut encoder = vec![Vec::new(); shape.split];
        let crossing = DsCudaExecutor::encode_half(
            enc,
            c,
            shape,
            &plan,
            &plan_rows,
            &tokens,
            &ids,
            &to_encoder,
            &mut |l, n: &HalfNet| {
                encoder[l] = n.s.x.read::<u16>(n.dev.as_ref(), 0, count * width)?;
                Ok(())
            },
        )?;
        assert_eq!(us(crossing.rows), count, "every position's row crosses");
        let encode = at.elapsed().as_secs_f64();

        // The decoder reads every position's logits.
        let at = Instant::now();
        let (_, mut dplan) = self.plan(from, to);
        for it in &mut dplan.items {
            it.top = true;
        }
        dplan.top_rows = n32(count);
        assert_eq!(
            dplan
                .items
                .iter()
                .map(|it| (it.slot, it.pos))
                .collect::<Vec<_>>(),
            crossing.positions,
            "the head's rows keep the crossing's order"
        );
        crossing.check(&dplan, width, c.hc_mult, 0)?;
        let split = shape.split;
        let mut decoder = vec![Vec::new(); shape.depth()];
        let top = DsCudaExecutor::decode_half(
            dec,
            c,
            shape,
            &dplan,
            &crossing,
            &mut |l, n: &HalfNet| {
                decoder[l - split] = n.s.x.read::<u16>(n.dev.as_ref(), 0, count * width)?;
                Ok(())
            },
        )?;
        assert_eq!(us(top), count);
        let logits = dec
            .s
            .logits
            .read::<f32>(dec.dev.as_ref(), 0, count * c.vocab_size)?;
        let status = dec.status()?;
        if status != 0 {
            return Err(fail(format!("the decoder raised status {status:#x}")));
        }
        let decode = at.elapsed().as_secs_f64();
        Ok(Step {
            items,
            encoder,
            crossing,
            decoder,
            logits,
            seconds: [gather, encode, decode, began.elapsed().as_secs_f64()],
        })
    }

    /// The whole sequence through both halves, chunk by chunk, `watch` recording what
    /// each step did beside its times.
    pub fn run<W: Watch>(
        &self,
        enc: &mut HalfNet,
        dec: &mut HalfNet,
        watch: &W,
    ) -> Result<Recording> {
        let (split, depth) = (self.l.shape.split, self.l.shape.depth());
        let mut r = Recording {
            encoder: vec![Vec::new(); split],
            entries: Vec::new(),
            streams: Vec::new(),
            pre: Vec::new(),
            decoder: vec![Vec::new(); depth],
            logits: Vec::new(),
            steps: Vec::new(),
        };
        for (from, to) in self.chunks() {
            let before = watch.snap(enc, dec);
            let s = self.step(enc, dec, from, to)?;
            assert!(s.items.iter().all(|(slot, _)| *slot == self.slot));
            for (all, layer) in r.encoder.iter_mut().zip(&s.encoder) {
                all.extend_from_slice(layer);
            }
            for (all, layer) in r.decoder.iter_mut().zip(&s.decoder) {
                all.extend_from_slice(layer);
            }
            r.entries.extend_from_slice(&s.crossing.entries);
            r.streams.extend_from_slice(&s.crossing.streams);
            r.pre.extend_from_slice(&s.crossing.pre_mix);
            r.logits.extend_from_slice(&s.logits);
            let after = watch.snap(enc, dec);
            eprintln!("{from}..{to}: {:.2} s", s.seconds[3]);
            let mut step = json!({
                "positions": to - from,
                "seconds": s.seconds[3], "gather_seconds": s.seconds[0],
                "encoder_seconds": s.seconds[1], "decoder_seconds": s.seconds[2],
            });
            let watched = watch.delta(&before, &after);
            if !watched.is_null() {
                step["cache"] = watched;
            }
            r.steps.push(step);
        }
        Ok(r)
    }
}

impl Recording {
    /// Writes the recording as `ds_halves` writes its own.
    pub fn write(&self, out: &Path) -> Result<()> {
        std::fs::create_dir_all(out).map_err(fail)?;
        let write = |name: &str, bytes: &[u8]| std::fs::write(out.join(name), bytes).map_err(fail);
        write("encoder_layers.bin", bytes_of(&self.encoder.concat()))?;
        write("boundary_entries.bin", &self.entries)?;
        write("boundary_streams.bin", bytes_of(&self.streams))?;
        write("boundary_pre.bin", bytes_of(&self.pre))?;
        write("decoder_layers.bin", bytes_of(&self.decoder.concat()))?;
        write("logits.bin", bytes_of(&self.logits))
    }

    /// Every array against `dir`'s recordings of the same `n` positions, bit for bit:
    /// each check, and whether all are equal.
    pub fn compare(&self, l: &Layout, n: usize, dir: &Path) -> Result<(Vec<Value>, bool)> {
        let read =
            |name: &str| std::fs::read(dir.join(name)).map_err(|e| fail(format!("{name}: {e}")));
        let (width, eb, vocab, split) = (l.width(), l.entry_bytes(), l.c.vocab_size, l.shape.split);
        let mut checks = Vec::new();
        let mut equal = true;
        let mut check = |name: String, got: Vec<u32>, want: Vec<u32>, row: usize| {
            let c = first_difference(&name, &got, &want, row);
            equal &= c["equal"] == json!(true);
            eprintln!("{c}");
            checks.push(c);
        };
        let layers = u16s(&read("encoder_layers.bin")?);
        for (i, got) in self.encoder.iter().enumerate() {
            let want = layers
                .get(i * n * width..(i + 1) * n * width)
                .unwrap_or(&[]);
            check(
                format!("streams after encoder layer {i}"),
                bf16_bits(got),
                bf16_bits(want),
                width,
            );
        }
        check(
            "bank entries".into(),
            self.entries.iter().map(|b| u32::from(*b)).collect(),
            read("boundary_entries.bin")?
                .iter()
                .map(|b| u32::from(*b))
                .collect(),
            eb,
        );
        check(
            "boundary streams".into(),
            bf16_bits(&self.streams),
            bf16_bits(&u16s(&read("boundary_streams.bin")?)),
            width,
        );
        check(
            "boundary collapse weights".into(),
            u32s(bytes_of(&self.pre)),
            u32s(&read("boundary_pre.bin")?),
            l.c.hc_mult,
        );
        let layers = u16s(&read("decoder_layers.bin")?);
        for (j, got) in self.decoder.iter().enumerate() {
            let want = layers
                .get(j * n * width..(j + 1) * n * width)
                .unwrap_or(&[]);
            check(
                format!("streams after decoder layer {}", split + j),
                bf16_bits(got),
                bf16_bits(want),
                width,
            );
        }
        check(
            "logits".into(),
            u32s(bytes_of(&self.logits)),
            u32s(&read("logits.bin")?),
            vocab,
        );
        Ok((checks, equal))
    }
}

/// A half's device bytes by kind.
pub fn memory_json(m: &MemoryBytes) -> Value {
    json!({
        "weights": m.weights, "kv": m.kv, "tables": m.tables, "state": m.state,
        "scratch": m.scratch, "total": m.total(),
    })
}

pub fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

pub fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub fn bf16_bits(bits: &[u16]) -> Vec<u32> {
    bits.iter().map(|b| u32::from(*b) << 16).collect()
}

/// Whether two arrays of rows of `row` elements are equal bit for bit, and where they
/// first differ.
pub fn first_difference(name: &str, got: &[u32], want: &[u32], row: usize) -> Value {
    if got.len() != want.len() {
        return json!({"what": name, "equal": false, "lengths": [got.len(), want.len()]});
    }
    let differing = got.iter().zip(want).filter(|(a, b)| a != b).count();
    let first = got.iter().zip(want).position(|(a, b)| a != b);
    json!({
        "what": name,
        "equal": differing == 0,
        "elements": got.len(),
        "differing": differing,
        "first": first.map(|at| json!({
            "position": at / row, "element": at % row,
            "got": format!("{:#010x}", got[at]), "want": format!("{:#010x}", want[at]),
        })),
    })
}

/// The text and the Engram map's file: the one named, else the checkpoint's pinned
/// golden, else any of its reference goldens (as `ds_halves`).
pub fn golden(dir: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("EIDOLA_DS_HALVES_GOLDEN") {
        return PathBuf::from(path);
    }
    let golden = dir.join("golden");
    let pinned = golden.join("reference-pinned.safetensors");
    if pinned.exists() {
        return pinned;
    }
    let mut others: Vec<PathBuf> = std::fs::read_dir(&golden)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f.starts_with("reference-") && f.ends_with(".safetensors"))
        })
        .collect();
    others.sort();
    others.into_iter().next().unwrap_or(pinned)
}

/// One chat prompt: its name and tokens.
pub struct Prompt {
    pub name: String,
    pub tokens: Vec<u32>,
}

/// The rendered prompts, the end-of-sequence ids, the tokenizer and the Engram map.
pub type Rendered = (
    Vec<Prompt>,
    Vec<u32>,
    eidola_engine_chat::Tokenizer,
    Vec<u32>,
);

/// The prompts of `ds_generate`'s file, rendered and tokenized as the node does
/// (thinking on, effort `high`), with the tokenizer of the checkpoint in `dir`.
pub fn prompts(dir: &Path, file: &Path) -> std::result::Result<Rendered, String> {
    use eidola_engine_chat::deepseek_v4_1::{self, EncodeOptions};
    use eidola_engine_chat::{ChatFormat, ChatInput, Tokenizer};
    let tokenizer = Tokenizer::for_format_from_model_dir(ChatFormat::DeepSeekV4_1, dir)
        .map_err(|e| e.to_string())?;
    let map = deepseek_v4_1::engram::compressed_vocab(&tokenizer)
        .map_err(|e| e.to_string())?
        .map;
    let text = std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let specs: Vec<Value> = serde_json::from_slice(&text).map_err(|e| e.to_string())?;
    let options = EncodeOptions::from_request(None, None).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for s in specs {
        let name = s["name"]
            .as_str()
            .ok_or("a prompt without a name")?
            .to_owned();
        let messages = serde_json::to_string(&s["messages"]).map_err(|e| e.to_string())?;
        let tools = s
            .get("tools")
            .map(|t| serde_json::to_string(t).expect("JSON"));
        let input = ChatInput::from_json(&messages, tools.as_deref()).map_err(|e| e.to_string())?;
        let prompt = deepseek_v4_1::encode(&input, options).map_err(|e| e.to_string())?;
        let tokens = tokenizer
            .encode_prompt(&prompt)
            .map_err(|e| e.to_string())?;
        out.push(Prompt { name, tokens });
    }
    let eos = tokenizer.eos_token_ids().to_vec();
    Ok((out, eos, tokenizer, map))
}

/// `ds_generate`'s prompts file, beside that example.
pub fn default_prompts_file() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eidola-server-engine/examples/ds_generate/prompts.json"
    ))
}
