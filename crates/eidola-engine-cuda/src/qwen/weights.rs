//! The model on a device, in the kernels' layouts.
//!
//! Copied as stored, with four changes of form, each exact: FP8 block scales are BF16 in
//! the checkpoint and become f32 transposed to `[k / 128][n / 128]` (the CUTLASS SFB
//! layout both GEMM forms read); the FP8 projections of one input are stacked into one
//! weight, their rows one after another (each part a whole number of 128-row blocks, so
//! its outputs are a run of the stack's columns: the DeltaNet's `qkv` and `z`, the
//! attention's `q`, `k` and `v`, the MLP's `gate` and `up`), so a layer quantises each
//! input once and runs one GEMM over it, each output a function of its own weight row as
//! before; norm weights are the model crate's f32 (`1 + w` where
//! zero-centred); the DeltaNet's `a` and `b` projections, held by the model crate in f32,
//! go back to their BF16 bits, stacked in one matrix (`a`'s rows, `b`'s, then zero rows to
//! a multiple of 8: [`ab_rows`]) that the exact profile reads as two and the fast profile
//! multiplies at once. `exp(A_log)` is computed once per head with the model
//! crate's `exp32`, and the RoPE tables are the model crate's `rope_cos_sin` at every
//! position the executor can rotate.

use std::any::Any;

use eidola_engine_model::Matrix;
use eidola_engine_model::qwen::QwenModel;
use eidola_engine_model::qwen::attention::rope_cos_sin;
use eidola_engine_model::qwen::linear::{Bf16Matrix, Fp8Linear};
use eidola_engine_model::qwen::numeric::exp32;
use eidola_engine_model::qwen::weights::{LayerWeights, MixerWeights, ZcNorm};

use super::device::QwenDevice;
use crate::{CudaError, Result, narrow};

/// An FP8 weight `[n, k]`: codes and f32 scales `[k / 128][n / 128]`.
#[derive(Clone, Copy, Debug)]
pub struct Fp8W {
    pub codes: u64,
    pub scales: u64,
    pub n: u32,
    pub k: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DeltaNetW {
    /// `in_proj_qkv`'s `conv_dim` rows, then `in_proj_z`'s `value_dim`.
    pub qkvz: Fp8W,
    /// BF16 `[ab_n, hidden]`: `a`'s `v_heads` rows, then `b`'s, then zero rows.
    pub ab: u64,
    /// [`ab_rows`] of the layer's value heads.
    pub ab_n: u32,
    /// `a` and `b` alone, BF16 `[v_heads, hidden]`: rows of `ab`.
    pub a: u64,
    pub b: u64,
    /// f32 `[conv_dim, 4]`.
    pub conv: u64,
    /// f32 `[v_heads]`: `exp32(A_log)`.
    pub exp_a_log: u64,
    pub dt_bias: u64,
    /// f32 `[v_dim]`, the plain gated-norm weight.
    pub norm: u64,
    pub out: Fp8W,
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionW {
    /// `q_proj`'s `2 · q_heads · head_dim` rows (per head its query then its gate), then
    /// `k_proj`'s and `v_proj`'s `kv_heads · head_dim` each.
    pub qkv: Fp8W,
    pub o: Fp8W,
    pub q_norm: u64,
    pub k_norm: u64,
}

#[derive(Clone, Copy, Debug)]
pub enum MixerW {
    DeltaNet(DeltaNetW),
    Attention(AttentionW),
}

#[derive(Clone, Copy, Debug)]
pub struct LayerW {
    pub input_norm: u64,
    pub post_norm: u64,
    pub mixer: MixerW,
    /// `gate_proj`'s `intermediate` rows, then `up_proj`'s.
    pub gate_up: Fp8W,
    pub down: Fp8W,
}

#[derive(Clone, Copy, Debug)]
pub struct MtpW {
    pub pre_fc_norm_embedding: u64,
    pub pre_fc_norm_hidden: u64,
    /// BF16 `[hidden, 2 · hidden]`.
    pub fc: u64,
    pub layer: LayerW,
    pub norm: u64,
}

/// Rows of the stacked `a` and `b` projections: `2 · v_heads` rounded up to a multiple of
/// 8, the BF16 GEMM's alignment (96 on the released model, which needs no zero rows).
pub fn ab_rows(v_heads: u32) -> u32 {
    (2 * v_heads).next_multiple_of(8)
}

/// Every weight of the forward on one device.
pub struct DevWeights {
    /// BF16 `[vocab, hidden]`.
    pub embed: u64,
    pub head: u64,
    pub final_norm: u64,
    pub layers: Vec<LayerW>,
    pub mtp: Option<MtpW>,
    /// f32 `[positions, rotary]`.
    pub rope_cos: u64,
    pub rope_sin: u64,
    pub rope_positions: u32,
    /// Bytes on the device.
    pub bytes: usize,
    keep: Vec<Box<dyn Any>>,
}

struct Loader<'d, D: QwenDevice> {
    dev: &'d D,
    keep: Vec<Box<dyn Any>>,
    bytes: usize,
}

impl<D: QwenDevice> Loader<'_, D> {
    fn put(&mut self, data: &[u8]) -> Result<u64> {
        let (addr, keep) = self.dev.alloc(data.len().max(8))?;
        // SAFETY: a fresh allocation of at least `data.len()` bytes.
        unsafe { self.dev.upload(addr, data)? };
        self.keep.push(keep);
        self.bytes += data.len();
        Ok(addr)
    }

    fn f32s(&mut self, x: &[f32]) -> Result<u64> {
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.put(&bytes)
    }

    fn bf16_matrix(&mut self, m: &Bf16Matrix) -> Result<u64> {
        self.put(m.data.bytes())
    }

    /// `a` and `b` (f32 matrices of BF16 values, `[v_heads, k]` each) stacked as BF16
    /// bits, zero rows to [`ab_rows`]; the stack's address and rows.
    fn ab(&mut self, a: &Matrix, b: &Matrix) -> Result<(u64, u32)> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(CudaError::new(
                "the DeltaNet's a and b projections differ in shape",
            ));
        }
        let n = ab_rows(narrow(a.rows, "DeltaNet value heads")?);
        let mut bytes = Vec::with_capacity(n as usize * a.cols * 2);
        for m in [a, b] {
            for &v in &m.data {
                let x = v.to_bits();
                if x & 0xffff != 0 {
                    return Err(CudaError::new("a BF16 weight that is not a BF16 value"));
                }
                bytes.extend(((x >> 16) as u16).to_le_bytes());
            }
        }
        bytes.resize(n as usize * a.cols * 2, 0);
        Ok((self.put(&bytes)?, n))
    }

    fn norm(&mut self, n: &ZcNorm) -> Result<u64> {
        self.f32s(&n.0)
    }

    fn fp8(&mut self, w: &Fp8Linear) -> Result<Fp8W> {
        self.stack(&[w])
    }

    /// `parts` stacked by rows into one weight: the codes one after another, and the
    /// scales transposed to `[k / 128][n / 128]` over the stack's blocks.
    fn stack(&mut self, parts: &[&Fp8Linear]) -> Result<Fp8W> {
        let k = parts[0].cols;
        if parts.iter().any(|w| w.cols != k) {
            return Err(CudaError::new("stacked FP8 weights differ in K"));
        }
        let rows: usize = parts.iter().map(|w| w.rows).sum();
        let mut codes = Vec::with_capacity(rows * k);
        for w in parts {
            let c = w.codes.bytes();
            if let Some(i) = c.iter().position(|&c| c & 0x7f == 0x7f) {
                return Err(CudaError::new(format!(
                    "an FP8 weight holds a NaN code at element {i}"
                )));
            }
            codes.extend_from_slice(c);
        }
        let (nb, kb) = (rows / 128, k / 128);
        let mut t = vec![0.0f32; nb * kb];
        let mut first = 0;
        for w in parts {
            for i in 0..w.rows / 128 {
                for j in 0..kb {
                    t[j * nb + first + i] = w.scales[i * kb + j];
                }
            }
            first += w.rows / 128;
        }
        Ok(Fp8W {
            codes: self.put(&codes)?,
            scales: self.f32s(&t)?,
            n: narrow(rows, "FP8 rows")?,
            k: narrow(k, "FP8 columns")?,
        })
    }

    fn layer(&mut self, l: &LayerWeights) -> Result<LayerW> {
        let mixer = match &l.mixer {
            MixerWeights::DeltaNet(d) => {
                let exp_a: Vec<f32> = d.a_log.iter().map(|&x| exp32(x)).collect();
                let (ab, ab_n) = self.ab(&d.in_proj_a, &d.in_proj_b)?;
                MixerW::DeltaNet(DeltaNetW {
                    qkvz: self.stack(&[&d.in_proj_qkv, &d.in_proj_z])?,
                    ab,
                    ab_n,
                    a: ab,
                    b: ab + (d.in_proj_a.rows * d.in_proj_a.cols * 2) as u64,
                    conv: self.f32s(&d.conv)?,
                    exp_a_log: self.f32s(&exp_a)?,
                    dt_bias: self.f32s(&d.dt_bias)?,
                    norm: self.f32s(&d.norm)?,
                    out: self.fp8(&d.out_proj)?,
                })
            }
            MixerWeights::Attention(a) => MixerW::Attention(AttentionW {
                qkv: self.stack(&[&a.q_proj, &a.k_proj, &a.v_proj])?,
                o: self.fp8(&a.o_proj)?,
                q_norm: self.norm(&a.q_norm)?,
                k_norm: self.norm(&a.k_norm)?,
            }),
        };
        Ok(LayerW {
            input_norm: self.norm(&l.input_norm)?,
            post_norm: self.norm(&l.post_norm)?,
            mixer,
            gate_up: self.stack(&[&l.mlp.gate, &l.mlp.up])?,
            down: self.fp8(&l.mlp.down)?,
        })
    }
}

impl DevWeights {
    /// The model's weights on `dev`, the drafter's when `drafting`, and RoPE tables for
    /// positions `0 .. rope_positions`.
    pub fn load<D: QwenDevice>(
        dev: &D,
        model: &QwenModel,
        drafting: bool,
        rope_positions: u32,
    ) -> Result<DevWeights> {
        let w = &model.weights;
        let mut ld = Loader {
            dev,
            keep: Vec::new(),
            bytes: 0,
        };
        let embed = ld.bf16_matrix(&w.embed)?;
        let head = ld.bf16_matrix(&w.head)?;
        let final_norm = ld.norm(&w.final_norm)?;
        let layers = w
            .layers
            .iter()
            .map(|l| ld.layer(l))
            .collect::<Result<Vec<_>>>()?;
        let mtp = if drafting {
            let m = w
                .mtp
                .first()
                .ok_or_else(|| CudaError::new("drafting needs the model's MTP layer"))?;
            if !matches!(m.layer.mixer, MixerWeights::Attention(_)) {
                return Err(CudaError::new("the MTP layer is not an attention layer"));
            }
            Some(MtpW {
                pre_fc_norm_embedding: ld.norm(&m.pre_fc_norm_embedding)?,
                pre_fc_norm_hidden: ld.norm(&m.pre_fc_norm_hidden)?,
                fc: ld.bf16_matrix(&m.fc)?,
                layer: ld.layer(&m.layer)?,
                norm: ld.norm(&m.norm)?,
            })
        } else {
            None
        };
        let rot = model.config().attention.rotary_dim;
        let mut cos = Vec::with_capacity(rope_positions as usize * rot);
        let mut sin = Vec::with_capacity(rope_positions as usize * rot);
        let freq = eidola_engine_model::qwen::attention::inv_freq(&model.config().attention);
        for pos in 0..rope_positions as usize {
            let (c, s) = rope_cos_sin(&freq, pos);
            cos.extend(c);
            sin.extend(s);
        }
        let rope_cos = ld.f32s(&cos)?;
        let rope_sin = ld.f32s(&sin)?;
        Ok(DevWeights {
            embed,
            head,
            final_norm,
            layers,
            mtp,
            rope_cos,
            rope_sin,
            rope_positions,
            bytes: ld.bytes,
            keep: ld.keep,
        })
    }

    /// Keeps the allocations alive (and lets the executor drop them with itself).
    pub fn allocations(&self) -> usize {
        self.keep.len()
    }
}
