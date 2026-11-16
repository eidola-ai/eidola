//! Checkpoint tensors → the DeepSeek-V4.1 reference weights.
//!
//! Dense tensors are decoded once at load. Routed experts stay in the
//! memory-mapped files and are decoded per forward call; Engram tables are
//! read one row at a time. Vision tensors and the image router bias
//! (`gate.bias_vl`) are never read.

use std::sync::Arc;

use super::config::{DsConfig, DsLayerSpec, DsMoeSpec};
use super::engram::{EngramTable, EngramWeights, MappedEngramTable};
use super::linear::{Fp4Linear, Fp8Linear};
use super::mhc::MhcWeights;
use super::moe::{Fp4Expert, Fp8Expert};
use super::numeric::bf16_round;
use crate::error::{Error, Result};
use crate::safetensors::{Dtype, WeightSet};
use crate::tensor::Matrix;

#[derive(Clone, Default)]
pub struct DsLoadOptions {
    /// Load the draft blocks (`mtp.*`) when the config has a drafter.
    pub skip_drafter: bool,
    /// Engram tables to use instead of the checkpoint's, keyed by checkpoint
    /// layer: a test's sparse stand-ins.
    pub engram_tables: Vec<(usize, Arc<dyn EngramTable>)>,
}

/// The global-KV side of a KV-source layer.
pub struct CompressorWeights {
    /// `[head_dim, dim]`.
    pub wkv: Matrix,
    /// The pooling gate; present when the ratio is above 1.
    pub wgate: Option<Matrix>,
    pub norm: Vec<f32>,
    /// `[index_head_dim, head_dim]`: index keys from the latent.
    pub wk: Matrix,
    pub k_norm: Vec<f32>,
}

pub struct IndexerWeights {
    /// `[index_heads · index_head_dim, q_lora_rank]`.
    pub wq_b: Fp8Linear,
    /// `[index_heads, dim]`.
    pub weights_proj: Matrix,
}

pub struct DsAttentionWeights {
    pub wq_a: Fp8Linear,
    pub q_norm: Vec<f32>,
    pub wq_b: Fp8Linear,
    pub wkv: Fp8Linear,
    pub kv_norm: Vec<f32>,
    pub sink: Vec<f32>,
    /// `[o_groups · o_lora_rank, heads · head_dim / o_groups]`, dequantised to
    /// BF16 values as the reference's conversion does: row `g·rank + r`
    /// projects group `g`'s heads.
    pub wo_a: Matrix,
    pub wo_b: Fp8Linear,
    pub compressor: Option<CompressorWeights>,
    pub indexer: Option<IndexerWeights>,
}

pub struct DsMoeWeights {
    /// `[n_routed, dim]` (stored BF16, used in f32).
    pub router: Matrix,
    pub bias: Vec<f32>,
    pub shared: Fp8Expert,
    prefix: String,
}

pub struct DsBlockWeights {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub attention: DsAttentionWeights,
    pub moe: DsMoeWeights,
    pub hc_attn: MhcWeights,
    pub hc_ffn: MhcWeights,
    pub engram: Option<EngramWeights>,
}

pub struct DrafterWeights {
    pub blocks: Vec<DsBlockWeights>,
    /// `[dim, dim · target layers]`.
    pub main_proj: Fp8Linear,
    pub main_norm: Vec<f32>,
    pub norm: Vec<f32>,
    pub markov_embed: Matrix,
    pub markov_head: Matrix,
    /// `[dim + markov_rank]`.
    pub confidence: Vec<f32>,
}

pub struct DsWeights {
    pub config: DsConfig,
    pub embed: Matrix,
    pub head: Matrix,
    pub final_norm: Vec<f32>,
    pub layers: Vec<DsBlockWeights>,
    pub drafter: Option<DrafterWeights>,
    store: Arc<WeightSet>,
}

struct Loader<'a> {
    store: &'a Arc<WeightSet>,
    c: &'a DsConfig,
}

impl Loader<'_> {
    fn bf16_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
        let t = self.store.get(name)?;
        t.expect(name, Dtype::Bf16, &[rows, cols])?;
        Ok(Matrix::from_vec(rows, cols, t.to_f32(name)?))
    }

    fn vec(&self, name: &str, dtype: Dtype, n: usize) -> Result<Vec<f32>> {
        let t = self.store.get(name)?;
        t.expect(name, dtype, &[n])?;
        t.to_f32(name)
    }

    fn norm(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        self.vec(name, Dtype::Bf16, n)
    }

    fn fp8(&self, prefix: &str, rows: usize, cols: usize) -> Result<Fp8Linear> {
        let (wn, sn) = (format!("{prefix}.weight"), format!("{prefix}.scale"));
        let (w, s) = (self.store.get(&wn)?, self.store.get(&sn)?);
        w.expect(&wn, Dtype::F8E4M3, &[rows, cols])?;
        s.expect(&sn, Dtype::F8E8M0, &[rows.div_ceil(32), cols / 32])?;
        Ok(Fp8Linear::from_bytes(rows, cols, w.data, s.data))
    }

    fn mhc(&self, p: &str, sub: &str) -> Result<MhcWeights> {
        let (hc, d) = (self.c.hc_mult, self.c.dim);
        let n = (2 + hc) * hc;
        let name = format!("{p}.hc_{sub}_fn");
        let t = self.store.get(&name)?;
        t.expect(&name, Dtype::F32, &[n, hc * d])?;
        let scale = self.vec(&format!("{p}.hc_{sub}_scale"), Dtype::F32, 3)?;
        Ok(MhcWeights {
            projection: Matrix::from_vec(n, hc * d, t.to_f32(&name)?),
            base: self.vec(&format!("{p}.hc_{sub}_base"), Dtype::F32, n)?,
            scale: [scale[0], scale[1], scale[2]],
        })
    }

    fn block(
        &self,
        p: &str,
        moe: &DsMoeSpec,
        layer: Option<&DsLayerSpec>,
        opts: &DsLoadOptions,
    ) -> Result<DsBlockWeights> {
        let c = self.c;
        let (d, a) = (c.dim, &c.attention);
        let group_width = a.n_heads * a.head_dim / a.o_groups;
        let mut compressor = None;
        let mut indexer = None;
        let mut engram = None;
        if let Some(l) = layer {
            if l.is_kv_source {
                let q = format!("{p}.attn.compressor");
                let i = format!("{p}.attn.indexer");
                compressor = Some(CompressorWeights {
                    wkv: self.bf16_matrix(&format!("{q}.wkv.weight"), a.head_dim, d)?,
                    wgate: match l.compress_ratio {
                        1 => None,
                        _ => Some(self.bf16_matrix(&format!("{q}.wgate.weight"), a.head_dim, d)?),
                    },
                    norm: self.norm(&format!("{q}.norm.weight"), a.head_dim)?,
                    wk: self.bf16_matrix(
                        &format!("{i}.wk.weight"),
                        c.indexer.head_dim,
                        a.head_dim,
                    )?,
                    k_norm: self.norm(&format!("{i}.k_norm.weight"), c.indexer.head_dim)?,
                });
            }
            if l.is_index_source {
                let i = format!("{p}.attn.indexer");
                indexer = Some(IndexerWeights {
                    wq_b: self.fp8(
                        &format!("{i}.wq_b"),
                        c.indexer.n_heads * c.indexer.head_dim,
                        a.q_lora_rank,
                    )?,
                    weights_proj: self.bf16_matrix(
                        &format!("{i}.weights_proj.weight"),
                        c.indexer.n_heads,
                        d,
                    )?,
                });
            }
            if let (Some(slot), Some(spec)) = (l.engram, &c.engram) {
                let hash = &spec.layers[slot];
                let table: Arc<dyn EngramTable> = match opts
                    .engram_tables
                    .iter()
                    .find(|(s, _)| *s == l.source_index)
                {
                    Some((_, t)) => t.clone(),
                    None => Arc::new(MappedEngramTable::open(
                        self.store.clone(),
                        &format!("{p}.engram.embed"),
                        hash.num_rows,
                        spec.head_dim,
                    )?),
                };
                if table.num_rows() != hash.num_rows || table.dim() != spec.head_dim {
                    return Err(Error::layout(
                        &format!("{p}.engram.embed"),
                        "table shape does not match the config",
                    ));
                }
                let q = self.bf16_matrix(&format!("{p}.engram.q_weight"), c.hc_mult, d)?;
                let k = self.bf16_matrix(&format!("{p}.engram.k_weight"), c.hc_mult, d)?;
                let product = q.data.iter().zip(&k.data).map(|(a, b)| a * b).collect();
                engram = Some(EngramWeights {
                    table,
                    wkv: self.fp8(
                        &format!("{p}.engram.wkv"),
                        d * (c.hc_mult + 1),
                        spec.lookups() * spec.head_dim,
                    )?,
                    gate_weight: Matrix::from_vec(c.hc_mult, d, product),
                });
            }
        }
        let router_name = format!("{p}.ffn.gate.weight");
        let router = self.bf16_matrix(&router_name, moe.source_n_routed, d)?;
        let bias = self.vec(
            &format!("{p}.ffn.gate.bias"),
            Dtype::F32,
            moe.source_n_routed,
        )?;
        let s = format!("{p}.ffn.shared_experts");
        Ok(DsBlockWeights {
            attn_norm: self.norm(&format!("{p}.attn_norm.weight"), d)?,
            ffn_norm: self.norm(&format!("{p}.ffn_norm.weight"), d)?,
            attention: DsAttentionWeights {
                wq_a: self.fp8(&format!("{p}.attn.wq_a"), a.q_lora_rank, d)?,
                q_norm: self.norm(&format!("{p}.attn.q_norm.weight"), a.q_lora_rank)?,
                wq_b: self.fp8(
                    &format!("{p}.attn.wq_b"),
                    a.n_heads * a.head_dim,
                    a.q_lora_rank,
                )?,
                wkv: self.fp8(&format!("{p}.attn.wkv"), a.head_dim, d)?,
                kv_norm: self.norm(&format!("{p}.attn.kv_norm.weight"), a.head_dim)?,
                sink: self.vec(&format!("{p}.attn.attn_sink"), Dtype::F32, a.n_heads)?,
                wo_a: self
                    .fp8(
                        &format!("{p}.attn.wo_a"),
                        a.o_groups * a.o_lora_rank,
                        group_width,
                    )?
                    .dequantised_bf16(),
                wo_b: self.fp8(&format!("{p}.attn.wo_b"), d, a.o_groups * a.o_lora_rank)?,
                compressor,
                indexer,
            },
            moe: DsMoeWeights {
                router: Matrix::from_vec(moe.n_routed, d, router.data[..moe.n_routed * d].to_vec()),
                bias: bias[..moe.n_routed].to_vec(),
                shared: Fp8Expert {
                    w1: self.fp8(&format!("{s}.w1"), moe.inter_dim, d)?,
                    w2: self.fp8(&format!("{s}.w2"), d, moe.inter_dim)?,
                    w3: self.fp8(&format!("{s}.w3"), moe.inter_dim, d)?,
                },
                prefix: format!("{p}.ffn.experts"),
            },
            hc_attn: self.mhc(p, "attn")?,
            hc_ffn: self.mhc(p, "ffn")?,
            engram,
        })
    }
}

impl DsWeights {
    pub fn load(store: Arc<WeightSet>, config: DsConfig, opts: &DsLoadOptions) -> Result<Self> {
        let l = Loader {
            store: &store,
            c: &config,
        };
        let (v, d) = (config.vocab_size, config.dim);
        let embed = l.bf16_matrix("embed.weight", v, d)?;
        let head = l.bf16_matrix("head.weight", v, d)?;
        let final_norm = l.norm("norm.weight", d)?;
        let mut layers = Vec::with_capacity(config.layers.len());
        for spec in &config.layers {
            layers.push(l.block(
                &format!("layers.{}", spec.source_index),
                &config.moe,
                Some(spec),
                opts,
            )?);
        }
        let drafter = match &config.drafter {
            Some(spec) if !opts.skip_drafter => {
                let mut blocks = Vec::with_capacity(spec.blocks);
                for i in 0..spec.blocks {
                    blocks.push(l.block(&format!("mtp.{i}"), &spec.moe, None, opts)?);
                }
                let last = format!("mtp.{}", spec.blocks - 1);
                let rank = spec.markov_rank;
                let confidence =
                    l.bf16_matrix(&format!("{last}.confidence_head.proj.weight"), 1, d + rank)?;
                Some(DrafterWeights {
                    blocks,
                    main_proj: l.fp8("mtp.0.main_proj", d, d * spec.target_layers.len())?,
                    main_norm: l.norm("mtp.0.main_norm.weight", d)?,
                    norm: l.norm(&format!("{last}.norm.weight"), d)?,
                    markov_embed: l.bf16_matrix(
                        &format!("{last}.markov_head.embed.weight"),
                        v,
                        rank,
                    )?,
                    markov_head: l.bf16_matrix(
                        &format!("{last}.markov_head.head.weight"),
                        v,
                        rank,
                    )?,
                    confidence: confidence.data,
                })
            }
            _ => None,
        };
        Ok(DsWeights {
            config,
            embed,
            head,
            final_norm,
            layers,
            drafter,
            store,
        })
    }

    /// Decode one routed expert of a block from its packed FP4 storage.
    pub fn expert(&self, moe: &DsMoeWeights, spec: &DsMoeSpec, e: usize) -> Result<Fp4Expert> {
        let d = self.config.dim;
        let fp4 = |w: &str, rows: usize, cols: usize| -> Result<Fp4Linear> {
            let (wn, sn) = (
                format!("{}.{e}.{w}.weight", moe.prefix),
                format!("{}.{e}.{w}.scale", moe.prefix),
            );
            let (wt, st) = (self.store.get(&wn)?, self.store.get(&sn)?);
            wt.expect(&wn, Dtype::I8, &[rows, cols / 2])?;
            st.expect(&sn, Dtype::F8E8M0, &[rows, cols / 32])?;
            Ok(Fp4Linear::from_bytes(rows, cols, wt.data, st.data))
        };
        Ok(Fp4Expert {
            w1: fp4("w1", spec.inter_dim, d)?,
            w2: fp4("w2", d, spec.inter_dim)?,
            w3: fp4("w3", spec.inter_dim, d)?,
        })
    }
}

/// `weight · (x · rsqrt(mean(x²) + eps))`, rounded to BF16: the reference's
/// `RMSNorm` on a BF16 tensor.
pub fn rms(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    crate::tensor::rms_norm(x, weight, eps, &mut out);
    for v in &mut out {
        *v = bf16_round(*v);
    }
    out
}
