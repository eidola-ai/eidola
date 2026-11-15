//! One half's weights on its device, in the kernels' layouts, and their byte accounting.
//!
//! Tensors are copied as stored: FP8 linears as E4M3 codes with their E8M0 tile scales
//! (`wo_a` included), BF16 matrices as BF16, the hyper-connection projections as f32,
//! Engram tables as codes and scales. Three things change shape and nothing changes
//! value: a layer's routed experts are laid out expert-major (gate rows then up rows of
//! each expert in one array, the down rows in another, scales alike); norm weights are
//! widened from BF16 to the f32 the norm kernel reads; and an Engram module's two gate
//! weights are kept as their f32 product, the only form the forward uses. The image
//! router bias is not read. An FP8 weight holding a NaN code is refused: the block
//! products are defined over finite codes.
//!
//! [`weight_bytes`] is the same sum as pure arithmetic over the configuration, in the
//! checkpoint's form or the device's.

use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::DsConfig;
use eidola_engine_model::deepseek::config::{DsLayerSpec, DsMoeSpec, RopeKind};
use eidola_engine_model::deepseek::rope::{RopeTable, Yarn};
use eidola_engine_model::safetensors::Dtype;

use super::half::{Buffer, Device, Kind, Plain};
use super::support::DsShape;
use crate::{CudaError, Result};

/// Which half of the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    /// Logical device 0: the embedding, the layers before the split, the split layer's
    /// bank stub, the first Engram table.
    Encoder,
    /// Logical device 1: the layers from the split on, the head, the later Engram tables.
    Decoder,
}

impl Half {
    pub fn device(self) -> u32 {
        match self {
            Half::Encoder => 0,
            Half::Decoder => 1,
        }
    }
}

/// Weight bytes by kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeightBytes {
    pub embed: u64,
    /// Routed experts: packed FP4 codes and their scales.
    pub experts: u64,
    /// FP8 linears of the layers: codes and tile scales.
    pub fp8: u64,
    /// BF16 tensors of the layers (and of the bank stub).
    pub bf16: u64,
    /// f32 tensors of the layers (and of the bank stub).
    pub f32: u64,
    pub engram_tables: u64,
    /// Engram projections and gate weights.
    pub engram_modules: u64,
    /// The head and the final norm.
    pub head: u64,
    /// RoPE tables (device form only).
    pub rope: u64,
    /// The drafter: its blocks, projection, norms and Markov head, and on a device the
    /// embedding copy its slots read.
    pub drafter: u64,
}

impl WeightBytes {
    pub fn total(&self) -> u64 {
        self.embed
            + self.layers()
            + self.engram_tables
            + self.engram_modules
            + self.head
            + self.rope
            + self.drafter
    }

    /// The layers' own tensors, Engram aside.
    pub fn layers(&self) -> u64 {
        self.experts + self.fp8 + self.bf16 + self.f32
    }
}

/// Which bytes [`weight_bytes`] counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// The tensors as the checkpoint stores them, every tensor of the half's layers
    /// (the image router bias and every stored router row included).
    Checkpoint,
    /// What [`HalfWeights::load`] puts on the device for `positions` positions.
    Device { positions: u32 },
}

fn fp8_bytes(n: usize, k: usize) -> u64 {
    (n * k + n.div_ceil(32) * (k / 32)) as u64
}

/// One block's tensors (a backbone layer's or a draft block's), Engram aside.
fn block_bytes(
    c: &DsConfig,
    m: &DsMoeSpec,
    kv_source: Option<bool>,
    index_source: bool,
    device: bool,
) -> WeightBytes {
    let (d, a, ix) = (c.dim, &c.attention, &c.indexer);
    let hc = c.hc_mult;
    let mixes = (2 + hc) * hc;
    let mut out = WeightBytes::default();
    // A norm weight: BF16 as stored, f32 on the device.
    let norm = |n: usize, out: &mut WeightBytes| {
        if device {
            out.f32 += (4 * n) as u64;
        } else {
            out.bf16 += (2 * n) as u64;
        }
    };
    let experts = if device {
        m.n_routed
    } else {
        m.source_n_routed
    };
    out.f32 += (4 * a.n_heads + 4 * experts) as u64;
    if !device {
        // The image router bias, which nothing reads.
        out.f32 += (4 * experts) as u64;
    }
    out.f32 += 2 * (4 * mixes * hc * d + 4 * mixes + 12) as u64;
    for n in [d, d, a.q_lora_rank, a.head_dim] {
        norm(n, &mut out);
    }
    out.bf16 += (2 * experts * d) as u64;
    if let Some(gated) = kv_source {
        norm(a.head_dim, &mut out);
        norm(ix.head_dim, &mut out);
        out.bf16 += (2 * a.head_dim * d * (1 + usize::from(gated))) as u64;
        out.bf16 += (2 * ix.head_dim * a.head_dim) as u64;
    }
    let rank = a.o_groups * a.o_lora_rank;
    out.fp8 += fp8_bytes(a.q_lora_rank, d)
        + fp8_bytes(a.n_heads * a.head_dim, a.q_lora_rank)
        + fp8_bytes(a.head_dim, d)
        + fp8_bytes(rank, a.n_heads * a.head_dim / a.o_groups)
        + fp8_bytes(d, rank)
        + 2 * fp8_bytes(m.inter_dim, d)
        + fp8_bytes(d, m.inter_dim);
    if index_source {
        out.bf16 += (2 * ix.n_heads * d) as u64;
        out.fp8 += fp8_bytes(ix.n_heads * ix.head_dim, a.q_lora_rank);
    }
    let per_expert = 2 * (m.inter_dim * d / 2 + m.inter_dim * d / 32)
        + d * m.inter_dim / 2
        + d * m.inter_dim / 32;
    out.experts += (experts * per_expert) as u64;
    out
}

/// The weight bytes of `half`, by kind. The drafter is counted when `shape` drafts (a
/// device form) or, in the checkpoint's form, whenever the configuration has one.
pub fn weight_bytes(c: &DsConfig, shape: &DsShape, half: Half, form: Form) -> WeightBytes {
    let device = matches!(form, Form::Device { .. });
    let (d, hc) = (c.dim, c.hc_mult);
    let mut out = WeightBytes::default();
    let layers = match half {
        Half::Encoder => 0..shape.split,
        Half::Decoder => shape.split..shape.layers,
    };
    for l in layers {
        let spec = &c.layers[l];
        let b = block_bytes(
            c,
            &c.moe,
            spec.is_kv_source.then_some(spec.compress_ratio > 1),
            spec.is_index_source,
            device,
        );
        out.experts += b.experts;
        out.fp8 += b.fp8;
        out.bf16 += b.bf16;
        out.f32 += b.f32;
        if let (Some(_), Some(e)) = (spec.engram, &c.engram) {
            out.engram_modules +=
                fp8_bytes(d * (hc + 1), e.lookups() * e.head_dim) + (4 * hc * d) as u64;
        }
    }
    if let Some(e) = &c.engram {
        for (slot, l) in e.layers.iter().enumerate() {
            if shape.engram_device[slot] == half.device() && shape.engram_layer[slot].is_some() {
                out.engram_tables += l.num_rows * (e.head_dim + e.head_dim / 32) as u64;
            }
        }
    }
    let norm = |n: usize| if device { 4 * n } else { 2 * n } as u64;
    match half {
        Half::Encoder => {
            out.embed = (2 * c.vocab_size * d) as u64;
            if device {
                // The split layer's bank stub: its attention-input norm and compressor.
                let (a, ix) = (&c.attention, &c.indexer);
                out.f32 += norm(d) + norm(a.head_dim) + norm(ix.head_dim);
                out.bf16 += (2 * a.head_dim * d + 2 * ix.head_dim * a.head_dim) as u64;
            }
        }
        Half::Decoder => {
            out.head = (2 * c.vocab_size * d) as u64 + norm(d);
            let counted = if device { shape.drafting() } else { true };
            if let (Some(dr), true) = (&c.drafter, counted) {
                let b = block_bytes(c, &dr.moe, None, false, device);
                out.drafter = dr.blocks as u64 * (b.experts + b.fp8 + b.bf16 + b.f32)
                    + fp8_bytes(d, d * dr.target_layers.len())
                    + 2 * norm(d)
                    + 2 * (2 * c.vocab_size * dr.markov_rank) as u64;
                out.drafter += match device {
                    // The embedding rows the slots read: a copy of the table.
                    true => (2 * c.vocab_size * d) as u64,
                    // The confidence head, which nothing here reads.
                    false => (2 * (d + dr.markov_rank)) as u64,
                };
            }
        }
    }
    if let Form::Device { positions } = form {
        let kinds = rope_kinds(c, shape, half);
        out.rope = (kinds.len() * positions as usize * c.attention.rope_dim * 4) as u64;
    }
    out
}

/// The RoPE variants `half` rotates with.
fn rope_kinds(c: &DsConfig, shape: &DsShape, half: Half) -> Vec<RopeKind> {
    let layers = match half {
        // The stub rotates the split layer's entries.
        Half::Encoder => 0..shape.split + 1,
        Half::Decoder => shape.split..shape.layers,
    };
    let mut kinds = Vec::new();
    for l in layers {
        if !kinds.contains(&c.layers[l].rope) {
            kinds.push(c.layers[l].rope);
        }
    }
    // The drafter rotates with the plain table.
    if half == Half::Decoder && shape.drafting() && !kinds.contains(&RopeKind::Plain) {
        kinds.push(RopeKind::Plain);
    }
    kinds
}

/// An FP8 linear on the device.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp8 {
    /// `[n, k]` E4M3.
    pub codes: u64,
    /// `[ceil(n / 32), k / 32]` E8M0.
    pub scales: u64,
    pub n: u32,
    pub k: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Mhc {
    pub projection: u64,
    pub base: u64,
    pub scale: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct Compressor {
    /// BF16 `[head_dim, dim]`.
    pub wkv: u64,
    /// BF16 `[head_dim, dim]`, for a pooled source.
    pub wgate: Option<u64>,
    pub norm: u64,
    /// BF16 `[index_head_dim, head_dim]`.
    pub wk: u64,
    pub k_norm: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Indexer {
    pub wq_b: Fp8,
    /// BF16 `[index_heads, dim]`.
    pub weights_proj: u64,
}

/// A layer's routed experts, expert-major.
#[derive(Clone, Copy, Debug)]
pub struct Experts {
    /// `[experts, 2 · inter, dim / 2]`: each expert's gate rows, then its up rows.
    pub up_packed: u64,
    /// `[experts, 2 · inter, dim / 32]`.
    pub up_scales: u64,
    /// `[experts, dim, inter / 2]`.
    pub down_packed: u64,
    /// `[experts, dim, inter / 32]`.
    pub down_scales: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct EngramModule {
    /// Index in the config's Engram layers.
    pub slot: usize,
    pub wkv: Fp8,
    /// f32 `[copies, dim]`: `q_weight · k_weight`.
    pub gate_weight: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct EngramTable {
    pub codes: u64,
    pub scales: u64,
    pub rows: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct LayerWeights {
    pub attn_norm: u64,
    pub ffn_norm: u64,
    pub wq_a: Fp8,
    pub q_norm: u64,
    pub wq_b: Fp8,
    pub wkv: Fp8,
    pub kv_norm: u64,
    pub sink: u64,
    pub wo_a: Fp8,
    pub wo_b: Fp8,
    pub compressor: Option<Compressor>,
    pub indexer: Option<Indexer>,
    pub hc_attn: Mhc,
    pub hc_ffn: Mhc,
    /// BF16 `[experts, dim]`.
    pub router: u64,
    pub bias: u64,
    pub w1: Fp8,
    pub w2: Fp8,
    pub w3: Fp8,
    pub experts: Experts,
    pub engram: Option<EngramModule>,
}

/// The split layer's bank stub, on the encoder.
#[derive(Clone, Copy, Debug)]
pub struct Stub {
    pub attn_norm: u64,
    pub compressor: Compressor,
}

/// One half's weights.
pub struct HalfWeights {
    pub half: Half,
    /// By model layer; `None` for the other half's.
    pub layers: Vec<Option<LayerWeights>>,
    /// BF16 `[vocab, dim]` (encoder).
    pub embed: Option<u64>,
    /// BF16 `[vocab, dim]` and the final norm (decoder).
    pub head: Option<(u64, u64)>,
    pub stub: Option<Stub>,
    /// By Engram layer of the config: the tables this half holds.
    pub tables: Vec<Option<EngramTable>>,
    /// `(cos, sin)` tables `[positions, rope_dim / 2, 2]`, by variant.
    pub rope: Vec<(RopeKind, u64)>,
    pub positions: u32,
    /// The drafter (decoder, when drafting).
    pub drafter: Option<DrafterWeights>,
    /// What was uploaded, by kind.
    pub bytes: WeightBytes,
    _buffers: Vec<Buffer>,
}

/// The drafter on the decoder's device.
#[derive(Clone, Debug)]
pub struct DrafterWeights {
    pub blocks: Vec<LayerWeights>,
    /// `[dim, dim · target layers]`.
    pub main_proj: Fp8,
    pub main_norm: u64,
    /// The drafter's final norm.
    pub norm: u64,
    /// BF16 `[vocab, markov_rank]`, each.
    pub markov_embed: u64,
    pub markov_head: u64,
    /// BF16 `[vocab, dim]`: the embedding rows the slots read.
    pub embed: u64,
}

impl HalfWeights {
    pub fn rope(&self, kind: RopeKind) -> u64 {
        self.rope
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, at)| *at)
            .expect("a half loads the tables of its layers")
    }

    pub fn layer(&self, l: usize) -> &LayerWeights {
        self.layers[l].as_ref().expect("a layer of this half")
    }
}

/// The frequencies of a RoPE variant (`DsModel::rope`).
pub fn rope_table(c: &DsConfig, kind: RopeKind) -> RopeTable {
    let r = &c.rope;
    match kind {
        RopeKind::Plain => RopeTable::new(c.attention.rope_dim, r.theta, None),
        RopeKind::Compressed => RopeTable::new(
            c.attention.rope_dim,
            r.compress_theta,
            (r.original_seq_len > 0).then_some(Yarn {
                original_seq_len: r.original_seq_len,
                factor: r.factor,
                beta_fast: r.beta_fast,
                beta_slow: r.beta_slow,
            }),
        ),
    }
}

/// The first NaN among E4M3 codes: sign, an all-ones exponent and an all-ones mantissa.
pub fn first_nan_e4m3(codes: &[u8]) -> Option<usize> {
    codes.iter().position(|b| b & 0x7f == 0x7f)
}

fn model_err(e: eidola_engine_model::Error) -> CudaError {
    CudaError::new(e.to_string())
}

#[derive(Clone, Copy)]
enum Count {
    Embed,
    Experts,
    Fp8,
    Bf16,
    F32,
    Table,
    Module,
    Head,
    Rope,
}

struct Loader<'a> {
    dev: &'a dyn Device,
    /// The checkpoint, or none to lay the half out without its bytes.
    store: Option<&'a WeightSet>,
    buffers: Vec<Buffer>,
    bytes: WeightBytes,
    /// Count what is loaded as the drafter's.
    drafter: bool,
}

impl<'a> Loader<'a> {
    fn count(&mut self, kind: Count, bytes: usize) {
        let b = &mut self.bytes;
        if self.drafter {
            b.drafter += bytes as u64;
            return;
        }
        let field = match kind {
            Count::Embed => &mut b.embed,
            Count::Experts => &mut b.experts,
            Count::Fp8 => &mut b.fp8,
            Count::Bf16 => &mut b.bf16,
            Count::F32 => &mut b.f32,
            Count::Table => &mut b.engram_tables,
            Count::Module => &mut b.engram_modules,
            Count::Head => &mut b.head,
            Count::Rope => &mut b.rope,
        };
        *field += bytes as u64;
    }

    fn buffer(&mut self, kind: Count, bytes: usize) -> Result<&Buffer> {
        self.count(kind, bytes);
        self.buffers
            .push(Buffer::new(self.dev, "weights", Kind::Weights, bytes)?);
        Ok(self.buffers.last().expect("pushed"))
    }

    fn put<T: Plain>(&mut self, kind: Count, data: &[T]) -> Result<u64> {
        let dev = self.dev;
        let buf = self.buffer(kind, std::mem::size_of_val(data))?;
        buf.write(dev, 0, data)?;
        Ok(buf.addr)
    }

    /// A tensor's stored bytes, its dtype and shape checked.
    fn stored(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<Option<&'a [u8]>> {
        let Some(store) = self.store else {
            return Ok(None);
        };
        let t = store.get(name).map_err(model_err)?;
        t.expect(name, dtype, shape).map_err(model_err)?;
        Ok(Some(t.data))
    }

    /// A tensor as stored, or its first `take` bytes.
    fn raw(
        &mut self,
        kind: Count,
        name: &str,
        dtype: Dtype,
        shape: &[usize],
        take: Option<usize>,
    ) -> Result<u64> {
        let data = self.stored(name, dtype, shape)?;
        let len = take.unwrap_or(shape.iter().product::<usize>() * dtype.size());
        let dev = self.dev;
        let buf = self.buffer(kind, len)?;
        if let Some(data) = data {
            buf.write(dev, 0, &data[..len])?;
        }
        Ok(buf.addr)
    }

    fn values(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<Vec<f32>> {
        let Some(store) = self.store else {
            return Ok(vec![0.0; shape.iter().product()]);
        };
        let t = store.get(name).map_err(model_err)?;
        t.expect(name, dtype, shape).map_err(model_err)?;
        t.to_f32(name).map_err(model_err)
    }

    /// A BF16 norm weight, widened to f32.
    fn norm(&mut self, kind: Count, name: &str, n: usize) -> Result<u64> {
        let v = self.values(name, Dtype::Bf16, &[n])?;
        self.put(kind, &v)
    }

    fn fp8(&mut self, kind: Count, prefix: &str, n: usize, k: usize) -> Result<Fp8> {
        let (wn, sn) = (format!("{prefix}.weight"), format!("{prefix}.scale"));
        let codes = self.stored(&wn, Dtype::F8E4M3, &[n, k])?;
        if let Some(at) = codes.and_then(first_nan_e4m3) {
            return Err(CudaError::new(format!(
                "{wn}: element {at} is a NaN E4M3 code, which no block product is defined over"
            )));
        }
        let codes = self.raw(kind, &wn, Dtype::F8E4M3, &[n, k], None)?;
        let scales = self.raw(kind, &sn, Dtype::F8E8M0, &[n.div_ceil(32), k / 32], None)?;
        Ok(Fp8 {
            codes,
            scales,
            n: u32::try_from(n).map_err(|_| CudaError::new(format!("{wn}: rows")))?,
            k: u32::try_from(k).map_err(|_| CudaError::new(format!("{wn}: columns")))?,
        })
    }

    fn mhc(&mut self, c: &DsConfig, p: &str, sub: &str) -> Result<Mhc> {
        let n = (2 + c.hc_mult) * c.hc_mult;
        let projection = self.raw(
            Count::F32,
            &format!("{p}.hc_{sub}_fn"),
            Dtype::F32,
            &[n, c.hc_mult * c.dim],
            None,
        )?;
        let base = self.raw(
            Count::F32,
            &format!("{p}.hc_{sub}_base"),
            Dtype::F32,
            &[n],
            None,
        )?;
        let scale = self.values(&format!("{p}.hc_{sub}_scale"), Dtype::F32, &[3])?;
        // The three scales travel in the launch's struct; they are counted with the
        // layer's f32 tensors all the same.
        self.count(Count::F32, 12);
        Ok(Mhc {
            projection,
            base,
            scale: [scale[0], scale[1], scale[2]],
        })
    }

    fn compressor(&mut self, c: &DsConfig, p: &str, gated: bool) -> Result<Compressor> {
        let (d, hd, ihd) = (c.dim, c.attention.head_dim, c.indexer.head_dim);
        let (q, i) = (format!("{p}.attn.compressor"), format!("{p}.attn.indexer"));
        let bf16 = Dtype::Bf16;
        Ok(Compressor {
            wkv: self.raw(
                Count::Bf16,
                &format!("{q}.wkv.weight"),
                bf16,
                &[hd, d],
                None,
            )?,
            wgate: match gated {
                true => Some(self.raw(
                    Count::Bf16,
                    &format!("{q}.wgate.weight"),
                    bf16,
                    &[hd, d],
                    None,
                )?),
                false => None,
            },
            norm: self.norm(Count::F32, &format!("{q}.norm.weight"), hd)?,
            wk: self.raw(
                Count::Bf16,
                &format!("{i}.wk.weight"),
                bf16,
                &[ihd, hd],
                None,
            )?,
            k_norm: self.norm(Count::F32, &format!("{i}.k_norm.weight"), ihd)?,
        })
    }

    fn experts(&mut self, c: &DsConfig, p: &str, m: &DsMoeSpec) -> Result<Experts> {
        let d = c.dim;
        let (inter, n) = (m.inter_dim, m.n_routed);
        let dev = self.dev;
        // (per-expert bytes of the up array's two halves and of the down array)
        let up = [inter * d / 2, inter * d / 32];
        let down = [d * inter / 2, d * inter / 32];
        let sizes = [2 * up[0], 2 * up[1], down[0], down[1]];
        let mut addrs = [0u64; 4];
        let first = self.buffers.len();
        for (addr, bytes) in addrs.iter_mut().zip(sizes) {
            *addr = self.buffer(Count::Experts, n * bytes)?.addr;
        }
        for e in 0..n {
            let name = |w: &str, part: &str| format!("{p}.ffn.experts.{e}.{w}.{part}");
            let parts = [
                // (array, offset within the expert, tensor, dtype, shape)
                (0, 0, name("w1", "weight"), Dtype::I8, [inter, d / 2]),
                (0, up[0], name("w3", "weight"), Dtype::I8, [inter, d / 2]),
                (1, 0, name("w1", "scale"), Dtype::F8E8M0, [inter, d / 32]),
                (
                    1,
                    up[1],
                    name("w3", "scale"),
                    Dtype::F8E8M0,
                    [inter, d / 32],
                ),
                (2, 0, name("w2", "weight"), Dtype::I8, [d, inter / 2]),
                (3, 0, name("w2", "scale"), Dtype::F8E8M0, [d, inter / 32]),
            ];
            for (array, offset, tensor, dtype, shape) in parts {
                if let Some(data) = self.stored(&tensor, dtype, &shape)? {
                    self.buffers[first + array].write(dev, e * sizes[array] + offset, data)?;
                }
            }
        }
        Ok(Experts {
            up_packed: addrs[0],
            up_scales: addrs[1],
            down_packed: addrs[2],
            down_scales: addrs[3],
        })
    }

    /// A block's weights: a backbone layer's (with its sharing structure) or a draft
    /// block's.
    fn block(
        &mut self,
        c: &DsConfig,
        p: &str,
        m: &DsMoeSpec,
        spec: Option<&DsLayerSpec>,
    ) -> Result<LayerWeights> {
        let p = p.to_owned();
        let (d, a, ix) = (c.dim, &c.attention, &c.indexer);
        let rank = a.o_groups * a.o_lora_rank;
        let s = format!("{p}.ffn.shared_experts");
        let compressor = match spec {
            Some(l) if l.is_kv_source => Some(self.compressor(c, &p, l.compress_ratio > 1)?),
            _ => None,
        };
        let indexer = match spec.is_some_and(|l| l.is_index_source) {
            true => Some(Indexer {
                wq_b: self.fp8(
                    Count::Fp8,
                    &format!("{p}.attn.indexer.wq_b"),
                    ix.n_heads * ix.head_dim,
                    a.q_lora_rank,
                )?,
                weights_proj: self.raw(
                    Count::Bf16,
                    &format!("{p}.attn.indexer.weights_proj.weight"),
                    Dtype::Bf16,
                    &[ix.n_heads, d],
                    None,
                )?,
            }),
            false => None,
        };
        let engram = match (spec.and_then(|l| l.engram), &c.engram) {
            (Some(slot), Some(e)) => {
                let read =
                    |this: &Self, name: &str| this.values(name, Dtype::Bf16, &[c.hc_mult, d]);
                let q = read(self, &format!("{p}.engram.q_weight"))?;
                let k = read(self, &format!("{p}.engram.k_weight"))?;
                let product: Vec<f32> = q.iter().zip(&k).map(|(a, b)| a * b).collect();
                Some(EngramModule {
                    slot,
                    wkv: self.fp8(
                        Count::Module,
                        &format!("{p}.engram.wkv"),
                        d * (c.hc_mult + 1),
                        e.lookups() * e.head_dim,
                    )?,
                    gate_weight: self.put(Count::Module, &product)?,
                })
            }
            _ => None,
        };
        Ok(LayerWeights {
            attn_norm: self.norm(Count::F32, &format!("{p}.attn_norm.weight"), d)?,
            ffn_norm: self.norm(Count::F32, &format!("{p}.ffn_norm.weight"), d)?,
            wq_a: self.fp8(Count::Fp8, &format!("{p}.attn.wq_a"), a.q_lora_rank, d)?,
            q_norm: self.norm(
                Count::F32,
                &format!("{p}.attn.q_norm.weight"),
                a.q_lora_rank,
            )?,
            wq_b: self.fp8(
                Count::Fp8,
                &format!("{p}.attn.wq_b"),
                a.n_heads * a.head_dim,
                a.q_lora_rank,
            )?,
            wkv: self.fp8(Count::Fp8, &format!("{p}.attn.wkv"), a.head_dim, d)?,
            kv_norm: self.norm(Count::F32, &format!("{p}.attn.kv_norm.weight"), a.head_dim)?,
            sink: self.raw(
                Count::F32,
                &format!("{p}.attn.attn_sink"),
                Dtype::F32,
                &[a.n_heads],
                None,
            )?,
            wo_a: self.fp8(
                Count::Fp8,
                &format!("{p}.attn.wo_a"),
                rank,
                a.n_heads * a.head_dim / a.o_groups,
            )?,
            wo_b: self.fp8(Count::Fp8, &format!("{p}.attn.wo_b"), d, rank)?,
            compressor,
            indexer,
            hc_attn: self.mhc(c, &p, "attn")?,
            hc_ffn: self.mhc(c, &p, "ffn")?,
            router: self.raw(
                Count::Bf16,
                &format!("{p}.ffn.gate.weight"),
                Dtype::Bf16,
                &[m.source_n_routed, d],
                Some(2 * m.n_routed * d),
            )?,
            bias: self.raw(
                Count::F32,
                &format!("{p}.ffn.gate.bias"),
                Dtype::F32,
                &[m.source_n_routed],
                Some(4 * m.n_routed),
            )?,
            w1: self.fp8(Count::Fp8, &format!("{s}.w1"), m.inter_dim, d)?,
            w2: self.fp8(Count::Fp8, &format!("{s}.w2"), d, m.inter_dim)?,
            w3: self.fp8(Count::Fp8, &format!("{s}.w3"), m.inter_dim, d)?,
            experts: self.experts(c, &p, m)?,
            engram,
        })
    }
}

impl HalfWeights {
    /// Uploads `half`'s weights from `store`, with RoPE tables for `positions`
    /// positions. Without a store every tensor is allocated at its size and left zero:
    /// the half's layout and accounting with no checkpoint. Without `tables` the Engram tables the half's device would hold stay
    /// off it (a half run alone, whose table rows are gathered elsewhere).
    pub fn load(
        dev: &dyn Device,
        store: Option<&WeightSet>,
        c: &DsConfig,
        shape: &DsShape,
        half: Half,
        positions: u32,
        tables: bool,
    ) -> Result<HalfWeights> {
        let mut ld = Loader {
            dev,
            store,
            buffers: Vec::new(),
            bytes: WeightBytes::default(),
            drafter: false,
        };
        let (v, d) = (c.vocab_size, c.dim);
        let mut layers = vec![None; shape.layers];
        let range = match half {
            Half::Encoder => 0..shape.split,
            Half::Decoder => shape.split..shape.layers,
        };
        for l in range {
            let spec = &c.layers[l];
            layers[l] = Some(ld.block(
                c,
                &format!("layers.{}", spec.source_index),
                &c.moe,
                Some(spec),
            )?);
        }
        let load_tables = tables;
        let mut tables = vec![None; shape.engram_device.len()];
        if let (Some(e), true) = (&c.engram, load_tables) {
            for (slot, spec) in e.layers.iter().enumerate() {
                let Some(layer) = shape.engram_layer[slot] else {
                    continue;
                };
                if shape.engram_device[slot] != half.device() {
                    continue;
                }
                let p = format!("layers.{}.engram.embed", c.layers[layer].source_index);
                let rows = usize::try_from(spec.num_rows)
                    .map_err(|_| CudaError::new("Engram table rows"))?;
                tables[slot] = Some(EngramTable {
                    codes: ld.raw(
                        Count::Table,
                        &format!("{p}.weight"),
                        Dtype::F8E4M3,
                        &[rows, e.head_dim],
                        None,
                    )?,
                    scales: ld.raw(
                        Count::Table,
                        &format!("{p}.scale"),
                        Dtype::F8E8M0,
                        &[rows, e.head_dim / 32],
                        None,
                    )?,
                    rows: spec.num_rows,
                });
            }
        }
        let (mut embed, mut head, mut stub) = (None, None, None);
        match half {
            Half::Encoder => {
                embed = Some(ld.raw(Count::Embed, "embed.weight", Dtype::Bf16, &[v, d], None)?);
                let p = format!("layers.{}", c.layers[shape.split].source_index);
                stub = Some(Stub {
                    attn_norm: ld.norm(Count::F32, &format!("{p}.attn_norm.weight"), d)?,
                    compressor: ld.compressor(c, &p, false)?,
                });
            }
            Half::Decoder => {
                head = Some((
                    ld.raw(Count::Head, "head.weight", Dtype::Bf16, &[v, d], None)?,
                    ld.norm(Count::Head, "norm.weight", d)?,
                ));
            }
        }
        let mut drafter = None;
        if let (Half::Decoder, true, Some(dr)) = (half, shape.drafting(), &c.drafter) {
            ld.drafter = true;
            let mut blocks = Vec::with_capacity(dr.blocks);
            for i in 0..dr.blocks {
                blocks.push(ld.block(c, &format!("mtp.{i}"), &dr.moe, None)?);
            }
            let last = format!("mtp.{}", dr.blocks - 1);
            let rank = dr.markov_rank;
            let bf16 = Dtype::Bf16;
            drafter = Some(DrafterWeights {
                blocks,
                main_proj: ld.fp8(Count::Fp8, "mtp.0.main_proj", d, d * dr.target_layers.len())?,
                main_norm: ld.norm(Count::F32, "mtp.0.main_norm.weight", d)?,
                norm: ld.norm(Count::F32, &format!("{last}.norm.weight"), d)?,
                markov_embed: ld.raw(
                    Count::Bf16,
                    &format!("{last}.markov_head.embed.weight"),
                    bf16,
                    &[v, rank],
                    None,
                )?,
                markov_head: ld.raw(
                    Count::Bf16,
                    &format!("{last}.markov_head.head.weight"),
                    bf16,
                    &[v, rank],
                    None,
                )?,
                embed: ld.raw(Count::Embed, "embed.weight", bf16, &[v, d], None)?,
            });
            ld.drafter = false;
        }
        let mut rope = Vec::new();
        for kind in rope_kinds(c, shape, half) {
            let table = rope_table(c, kind);
            let mut cis = Vec::with_capacity(positions as usize * c.attention.rope_dim);
            for pos in 0..positions as usize {
                for (cos, sin) in table.cis(pos) {
                    cis.extend([cos, sin]);
                }
            }
            rope.push((kind, ld.put(Count::Rope, &cis)?));
        }
        Ok(HalfWeights {
            half,
            layers,
            embed,
            head,
            stub,
            tables,
            rope,
            positions,
            drafter,
            bytes: ld.bytes,
            _buffers: ld.buffers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::support::check_supported;
    use super::super::support::tests::{released_config, tiny_config};
    use super::*;

    /// The released checkpoint's totals, from its tensor headers: the two halves' layer
    /// tensors, both Engram tables and modules, the embedding, the head and the final
    /// norm. With the Engram tables and modules on the devices that hold them, that is
    /// 250,855,098,528 bytes of the checkpoint beside the encoder and 250,527,512,880
    /// beside the decoder before the drafter.
    #[test]
    fn the_released_checkpoint_as_arithmetic() {
        let c = released_config();
        let shape = check_supported(&c, 0).unwrap();
        let enc = weight_bytes(&c, &shape, Half::Encoder, Form::Checkpoint);
        let dec = weight_bytes(&c, &shape, Half::Decoder, Form::Checkpoint);
        assert_eq!(enc.embed, 1_323_827_200);
        assert_eq!(enc.layers(), 147_838_599_136);
        assert_eq!(dec.layers(), 147_823_271_392);
        assert_eq!(enc.engram_tables, 101_377_628_352);
        assert_eq!(dec.engram_tables, 101_380_404_048);
        assert_eq!(enc.engram_modules, 2 * 157_521_920);
        assert_eq!(dec.engram_modules, 0);
        assert_eq!(dec.head, 1_323_827_200 + 10_240);
        assert_eq!((enc.head, dec.embed, enc.rope, dec.rope), (0, 0, 0, 0));
        assert_eq!(enc.total(), 250_855_098_528);
        assert_eq!(dec.drafter, 7_932_874_632);
        assert_eq!(dec.total(), 250_527_512_880 + 7_932_874_632);
        // One routed expert: 18.8 MB; a layer's experts: 7.2 GB.
        assert_eq!(enc.experts, 20 * 384 * 18_800_640);

        // On the device: norms widen, the image router bias is not read, the encoder
        // also holds the split layer's bank stub, and each half its RoPE tables.
        let positions = 4096;
        let on = |half| weight_bytes(&c, &shape, half, Form::Device { positions });
        let (e, d) = (on(Half::Encoder), on(Half::Decoder));
        let norms = 2 * (5120 + 5120 + 1280 + 512) as u64;
        let compressor_norms = 2 * (512 + 128) as u64;
        let stub = 4 * 5120 + 4 * (512 + 128) + 2 * 512 * 5120 + 2 * 128 * 512;
        assert_eq!(
            e.layers(),
            enc.layers() + 20 * norms + 3 * compressor_norms - 20 * 1536 + stub
        );
        assert_eq!(
            d.layers(),
            dec.layers() + 20 * norms + compressor_norms - 20 * 1536
        );
        assert_eq!(d.head, dec.head + 10_240);
        // A table per RoPE variant the half's layers use, `(cos, sin)` per rotated pair.
        let table = 4096 * 64 * 4;
        assert_eq!((e.rope, d.rope), (2 * table, table));
        assert_eq!(
            (e.engram_tables, e.engram_modules),
            (enc.engram_tables, enc.engram_modules)
        );
    }

    #[test]
    fn nan_codes_are_the_two_all_ones_patterns() {
        use eidola_engine_model::numeric::fp8_e4m3_to_f32;
        for b in 0..=255u8 {
            assert_eq!(
                first_nan_e4m3(&[0, b]).is_some(),
                fp8_e4m3_to_f32(b).is_nan(),
                "{b:#x}"
            );
        }
        assert_eq!(first_nan_e4m3(&[1, 2, 0xff, 0x7f]), Some(2));
    }

    #[test]
    fn the_fixtures_halves() {
        let c = tiny_config();
        let shape = check_supported(&c, 0).unwrap();
        let ck = |half| weight_bytes(&c, &shape, half, Form::Checkpoint);
        // Layer 1's table beside the encoder, layer 2's beside the decoder; both
        // modules with their layers, on the encoder.
        assert_eq!(ck(Half::Encoder).engram_tables, 384 * 33);
        assert_eq!(ck(Half::Decoder).engram_tables, 552 * 33);
        assert_eq!(ck(Half::Decoder).engram_modules, 0);
        assert_eq!(ck(Half::Encoder).embed, 2 * 96 * 64);
    }
}
