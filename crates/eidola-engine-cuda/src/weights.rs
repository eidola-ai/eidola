//! The checkpoint on the device, in the layouts the kernels read.
//!
//! Tensors are copied from the memory-mapped safetensors as stored wherever a
//! kernel reads the checkpoint format directly (FP8 weights, MXFP4 expert
//! weights, BF16 matrices); only scale grids are re-laid-out, and small
//! vectors (norms, sinks, router bias) are widened to f32. Nothing is
//! dequantized.
//!
//! * **Fused QKV** stays rank-major: each of the `chunks` row blocks
//!   `[Q_c | K_c | V_c]` is padded to whole 128-row tiles, so the checkpoint's
//!   per-chunk scale grid (`ceil(R / 128)` scale rows per chunk) becomes an
//!   ordinary 128×128 grid. The RoPE/KV kernel reads Q, K and V out of the
//!   padded chunks.
//! * **FP8 scales** are transposed from `[N/128][K/128]` to the GEMM's
//!   `[K/128][N/128]`.
//! * **Dense gate/up** are stacked into one `[2I, H]` weight (one GEMM).
//! * **Experts**: gate and up rows stacked per expert (`[2I][H/2]`), all
//!   experts of a layer in one buffer; E8M0 scales repacked four per `i32`
//!   along K and transposed to `[K/128][N]` per expert (DeepGEMM's SFB).

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use eidola_engine_model::ModelConfig;
use eidola_engine_model::config::{AttentionSpec, FfnKind};
use eidola_engine_model::safetensors::{Dtype, TensorView, WeightSet};

use crate::model::Kernels;
use crate::support::check_supported;
use crate::{CudaError, Gpu, Result, narrow};

fn err(e: impl std::fmt::Display) -> CudaError {
    CudaError::new(e.to_string())
}

/// An FP8 block-scaled linear: `[n, k]` e4m3 and f32 scales `[k/128][n/128]`.
pub struct Fp8Linear {
    pub n: u32,
    pub k: u32,
    pub w: CudaSlice<u8>,
    pub scale: CudaSlice<f32>,
}

/// The fused QKV, padded per chunk.
pub struct QkvWeight {
    pub linear: Fp8Linear,
    pub chunks: u32,
    /// Padded rows per chunk (a multiple of 128).
    pub chunk_stride: u32,
    pub q_heads_per_chunk: u32,
    pub kv_heads_per_chunk: u32,
}

pub struct AttentionWeights {
    pub qkv: QkvWeight,
    /// BF16 `[H, nq·dv]`.
    pub o_proj: CudaSlice<u16>,
    /// f32 `[nq]`, `-inf` where the layer has no sinks.
    pub sinks: CudaSlice<f32>,
}

pub struct DenseFfn {
    /// `[2I, H]`: gate rows then up rows.
    pub gate_up: Fp8Linear,
    pub down: Fp8Linear,
    pub inter: u32,
}

pub struct MoeFfn {
    /// BF16 `[E, H]`.
    pub router: CudaSlice<u16>,
    pub bias: CudaSlice<f32>,
    /// `[E][2I][H/2]` packed E2M1.
    pub gate_up: CudaSlice<u8>,
    /// `[E][H/128][2I]` packed E8M0.
    pub gate_up_sf: CudaSlice<i32>,
    /// `[E][H][I/2]`.
    pub down: CudaSlice<u8>,
    /// `[E][I/128][H]`.
    pub down_sf: CudaSlice<i32>,
    pub experts: u32,
    pub top_k: u32,
    pub inter: u32,
    pub scaling: f32,
}

pub enum Ffn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

pub struct LayerWeights {
    pub input_norm: CudaSlice<f32>,
    pub post_attention_norm: CudaSlice<f32>,
    pub attention: AttentionWeights,
    pub ffn: Ffn,
}

pub struct ModelWeights {
    pub config: ModelConfig,
    /// BF16 `[V, H]`.
    pub embed: CudaSlice<u16>,
    pub lm_head: CudaSlice<u16>,
    pub final_norm: CudaSlice<f32>,
    pub layers: Vec<LayerWeights>,
    pub qkv_chunks: u32,
}

/// Checkpoint tensor families the executor does not serve, and leaves
/// unread: the vision and audio encoders (text-only serving) and the MTP
/// draft layers (no drafter yet).
pub const SKIPPED_PREFIXES: [&str; 4] = [
    "visual.",
    "audio_encoder.",
    "speech_embeddings.",
    "model.mtp.",
];

/// Every tensor the loader reads for `config`: the head, and each retained
/// layer's norms, attention (sinks exactly where the layer has them) and FFN.
pub fn expected_tensors(config: &ModelConfig) -> BTreeSet<String> {
    let mut t: BTreeSet<String> = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let fp8 = |t: &mut BTreeSet<String>, p: &str| {
        t.insert(format!("{p}.weight"));
        t.insert(format!("{p}.weight_scale_inv"));
    };
    for l in &config.layers {
        let p = format!("model.layers.{}", l.source_index);
        t.insert(format!("{p}.input_layernorm.weight"));
        t.insert(format!("{p}.post_attention_layernorm.weight"));
        fp8(&mut t, &format!("{p}.self_attn.qkv_proj"));
        t.insert(format!("{p}.self_attn.o_proj.weight"));
        if l.attention.has_sinks {
            t.insert(format!("{p}.self_attn.attention_sink_bias"));
        }
        match l.ffn {
            FfnKind::Dense => {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    fp8(&mut t, &format!("{p}.mlp.{proj}"));
                }
            }
            FfnKind::Moe => {
                t.insert(format!("{p}.mlp.gate.weight"));
                t.insert(format!("{p}.mlp.gate.e_score_correction_bias"));
                let experts = config.moe.as_ref().map_or(0, |m| m.num_experts);
                for x in 0..experts {
                    for proj in ["gate_proj", "up_proj", "down_proj"] {
                        t.insert(format!("{p}.mlp.experts.{x}.{proj}.weight"));
                        t.insert(format!("{p}.mlp.experts.{x}.{proj}.weight_scale"));
                    }
                }
            }
        }
    }
    t
}

/// The checkpoint holds exactly what the loader reads for `config`, besides
/// the families it skips ([`SKIPPED_PREFIXES`]) and the layers outside the
/// selection: a tensor the loader would not read (a sink on a layer without
/// sinks, an unknown module) and one it would read but cannot find are both
/// layout errors, refused before anything is read.
pub fn check_layout<'a>(names: impl Iterator<Item = &'a str>, config: &ModelConfig) -> Result<()> {
    let expected = expected_tensors(config);
    let retained: BTreeSet<usize> = config.layers.iter().map(|l| l.source_index).collect();
    let mut found = 0usize;
    for name in names {
        if expected.contains(name) {
            found += 1;
            continue;
        }
        if SKIPPED_PREFIXES.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        let layer = name
            .strip_prefix("model.layers.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(i, _)| i.parse::<usize>().ok());
        if layer.is_some_and(|i| i < config.source_num_layers && !retained.contains(&i)) {
            continue;
        }
        return Err(CudaError::new(format!(
            "checkpoint layout: {name} is not a tensor this configuration reads"
        )));
    }
    if found != expected.len() {
        return Err(CudaError::new(format!(
            "checkpoint layout: {} expected tensors are missing",
            expected.len() - found
        )));
    }
    Ok(())
}

struct Loader<'a> {
    gpu: &'a Gpu,
    store: &'a WeightSet,
    config: &'a ModelConfig,
    /// Every tensor name read, checked against [`expected_tensors`] at the
    /// end so the layout check and the loader cannot drift apart.
    read: RefCell<BTreeSet<String>>,
}

impl Loader<'_> {
    fn get(&self, name: &str) -> Result<TensorView<'_>> {
        self.read.borrow_mut().insert(name.to_owned());
        self.store.get(name).map_err(err)
    }

    /// A 1-D tensor of exactly `len` elements, widened to f32 (the kernels
    /// index it at the model's width, so a short one would read past it).
    fn f32_vec(&self, name: &str, len: usize) -> Result<CudaSlice<f32>> {
        let t = self.get(name)?;
        if t.shape != [len] {
            return Err(CudaError::new(format!(
                "{name}: shape {:?}, expected [{len}]",
                t.shape
            )));
        }
        let v = t.to_f32(name).map_err(err)?;
        Ok(self.gpu.stream().clone_htod(&v)?)
    }

    fn bf16(&self, name: &str, shape: &[usize]) -> Result<CudaSlice<u16>> {
        let t = self.get(name)?;
        t.expect(name, Dtype::Bf16, shape).map_err(err)?;
        let words: Vec<u16> = t
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect();
        Ok(self.gpu.stream().clone_htod(&words)?)
    }

    /// FP8 rows (with scale rows) of one weight.
    fn fp8_parts(&self, prefix: &str, n: usize, k: usize) -> Result<(&[u8], Vec<f32>, [usize; 2])> {
        let wname = format!("{prefix}.weight");
        let sname = format!("{prefix}.weight_scale_inv");
        let w = self.get(&wname)?;
        w.expect(&wname, Dtype::F8E4M3, &[n, k]).map_err(err)?;
        let s = self.get(&sname)?;
        if s.dtype != Dtype::F32 || s.shape.len() != 2 {
            return Err(CudaError::new(format!(
                "{sname}: {:?} {:?}",
                s.dtype, s.shape
            )));
        }
        let shape = [s.shape[0], s.shape[1]];
        Ok((w.data, s.to_f32(&sname).map_err(err)?, shape))
    }

    /// Several FP8 weights stacked along N (each a whole number of 128-row
    /// tiles), scales transposed to `[K/128][N/128]`.
    fn fp8_stacked(&self, prefixes: &[&str], n_each: usize, k: usize) -> Result<Fp8Linear> {
        if !n_each.is_multiple_of(128) || !k.is_multiple_of(128) {
            return Err(CudaError::new(format!(
                "{prefixes:?}: [{n_each}, {k}] is not tiled by 128"
            )));
        }
        let (nb, kb) = (n_each / 128, k / 128);
        let total_nb = nb * prefixes.len();
        let mut w = Vec::with_capacity(n_each * k * prefixes.len());
        let mut scale = vec![0f32; kb * total_nb];
        for (i, p) in prefixes.iter().enumerate() {
            let (data, s, shape) = self.fp8_parts(p, n_each, k)?;
            if shape != [nb, kb] {
                return Err(CudaError::new(format!("{p}: scale grid {shape:?}")));
            }
            w.extend_from_slice(data);
            for r in 0..nb {
                for c in 0..kb {
                    scale[c * total_nb + i * nb + r] = s[r * kb + c];
                }
            }
        }
        let st = self.gpu.stream();
        Ok(Fp8Linear {
            n: narrow(n_each * prefixes.len(), "GEMM rows")?,
            k: narrow(k, "GEMM depth")?,
            w: st.clone_htod(&w)?,
            scale: st.clone_htod(&scale)?,
        })
    }

    fn qkv(&self, prefix: &str, spec: &AttentionSpec, chunks: usize) -> Result<QkvWeight> {
        let h = self.config.hidden_size;
        let rows = spec.qkv_rows();
        if !rows.is_multiple_of(chunks)
            || !spec.num_q_heads.is_multiple_of(chunks)
            || !spec.num_kv_heads.is_multiple_of(chunks)
        {
            return Err(CudaError::new(format!(
                "{prefix}: QKV does not split into {chunks} chunks"
            )));
        }
        let r = rows / chunks;
        let tiles = r.div_ceil(128);
        let stride = tiles * 128;
        let (data, s, shape) = self.fp8_parts(&format!("{prefix}.self_attn.qkv_proj"), rows, h)?;
        let kb = h / 128;
        if shape != [chunks * tiles, kb] {
            return Err(CudaError::new(format!(
                "{prefix}: QKV scale grid {shape:?}"
            )));
        }
        let mut w = vec![0u8; chunks * stride * h];
        for c in 0..chunks {
            w[c * stride * h..(c * stride + r) * h]
                .copy_from_slice(&data[c * r * h..(c + 1) * r * h]);
        }
        let nb = chunks * tiles;
        let mut scale = vec![0f32; kb * nb];
        for row in 0..nb {
            for col in 0..kb {
                scale[col * nb + row] = s[row * kb + col];
            }
        }
        let st = self.gpu.stream();
        Ok(QkvWeight {
            linear: Fp8Linear {
                n: narrow(chunks * stride, "QKV rows")?,
                k: narrow(h, "hidden size")?,
                w: st.clone_htod(&w)?,
                scale: st.clone_htod(&scale)?,
            },
            chunks: narrow(chunks, "QKV chunks")?,
            chunk_stride: narrow(stride, "QKV chunk stride")?,
            q_heads_per_chunk: narrow(spec.num_q_heads / chunks, "query heads per chunk")?,
            kv_heads_per_chunk: narrow(spec.num_kv_heads / chunks, "KV heads per chunk")?,
        })
    }

    fn attention(
        &self,
        prefix: &str,
        spec: &AttentionSpec,
        chunks: usize,
    ) -> Result<AttentionWeights> {
        let h = self.config.hidden_size;
        let sinks = if spec.has_sinks {
            self.f32_vec(
                &format!("{prefix}.self_attn.attention_sink_bias"),
                spec.num_q_heads,
            )?
        } else {
            self.gpu
                .stream()
                .clone_htod(&vec![f32::NEG_INFINITY; spec.num_q_heads])?
        };
        Ok(AttentionWeights {
            qkv: self.qkv(prefix, spec, chunks)?,
            o_proj: self.bf16(
                &format!("{prefix}.self_attn.o_proj.weight"),
                &[h, spec.o_size()],
            )?,
            sinks,
        })
    }

    fn moe(&self, prefix: &str) -> Result<MoeFfn> {
        let spec = self
            .config
            .moe
            .as_ref()
            .ok_or_else(|| CudaError::new("no MoE config"))?;
        let (h, i, e) = (
            self.config.hidden_size,
            spec.intermediate_size,
            spec.num_experts,
        );
        let st = self.gpu.stream();
        let mut gate_up = st.alloc_zeros::<u8>(e * 2 * i * h / 2)?;
        let mut gate_up_sf = st.alloc_zeros::<i32>(e * (h / 128) * 2 * i)?;
        let mut down = st.alloc_zeros::<u8>(e * h * i / 2)?;
        let mut down_sf = st.alloc_zeros::<i32>(e * (i / 128) * h)?;
        // E8M0 [rows][cols/32] -> words [cols/128][rows].
        let words = |s: &[u8], rows: usize, cols: usize| -> Vec<i32> {
            let per = cols / 32;
            let mut out = vec![0i32; cols / 128 * rows];
            for r in 0..rows {
                for q in 0..cols / 128 {
                    let b = &s[r * per + 4 * q..r * per + 4 * q + 4];
                    out[q * rows + r] = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
            out
        };
        for x in 0..e {
            let p = format!("{prefix}.mlp.experts.{x}");
            let get = |proj: &str, rows: usize, cols: usize| -> Result<(&[u8], &[u8])> {
                let wn = format!("{p}.{proj}.weight");
                let sn = format!("{p}.{proj}.weight_scale");
                let w = self.get(&wn)?;
                w.expect(&wn, Dtype::U8, &[rows, cols / 2]).map_err(err)?;
                let s = self.get(&sn)?;
                s.expect(&sn, Dtype::U8, &[rows, cols / 32]).map_err(err)?;
                Ok((w.data, s.data))
            };
            let (gw, gs) = get("gate_proj", i, h)?;
            let (uw, us) = get("up_proj", i, h)?;
            let (dw, ds) = get("down_proj", h, i)?;
            let gu_bytes = 2 * i * h / 2;
            let mut v = gate_up.slice_mut(x * gu_bytes..x * gu_bytes + i * h / 2);
            st.memcpy_htod(gw, &mut v)?;
            let mut v = gate_up.slice_mut(x * gu_bytes + i * h / 2..(x + 1) * gu_bytes);
            st.memcpy_htod(uw, &mut v)?;
            let mut stacked = Vec::with_capacity(2 * i * h / 32);
            stacked.extend_from_slice(gs);
            stacked.extend_from_slice(us);
            let gsw = words(&stacked, 2 * i, h);
            let n = gsw.len();
            let mut v = gate_up_sf.slice_mut(x * n..(x + 1) * n);
            st.memcpy_htod(&gsw, &mut v)?;
            let d_bytes = h * i / 2;
            let mut v = down.slice_mut(x * d_bytes..(x + 1) * d_bytes);
            st.memcpy_htod(dw, &mut v)?;
            let dsw = words(ds, h, i);
            let n = dsw.len();
            let mut v = down_sf.slice_mut(x * n..(x + 1) * n);
            st.memcpy_htod(&dsw, &mut v)?;
        }
        Ok(MoeFfn {
            router: self.bf16(&format!("{prefix}.mlp.gate.weight"), &[e, h])?,
            bias: self.f32_vec(&format!("{prefix}.mlp.gate.e_score_correction_bias"), e)?,
            gate_up,
            gate_up_sf,
            down,
            down_sf,
            experts: narrow(e, "experts")?,
            top_k: narrow(spec.top_k, "experts per token")?,
            inter: narrow(i, "expert intermediate size")?,
            scaling: spec.routed_scaling_factor,
        })
    }
}

impl ModelWeights {
    /// Load `config`'s layers (a whole or truncated checkpoint) from `store`.
    /// Refuses (with [`CudaError::Unsupported`]) any configuration outside
    /// [`check_supported`] before reading a tensor. Taking the loaded kernels
    /// makes the order structural: no weight reaches the device until every
    /// kernel image has been read, verified against the manifest and loaded
    /// (a wrong device, image or kernel directory fails in seconds, not after
    /// the checkpoint has filled the GPU).
    pub fn load(
        gpu: &Gpu,
        _loaded: &Kernels,
        store: Arc<WeightSet>,
        config: ModelConfig,
    ) -> Result<ModelWeights> {
        check_supported(&config)?;
        check_layout(store.tensor_names(), &config)?;
        let chunks = crate::support::qkv_chunks(store.metadata("tp_size"), &config)?;
        let l = Loader {
            gpu,
            store: &store,
            config: &config,
            read: RefCell::new(BTreeSet::new()),
        };
        let (h, v) = (config.hidden_size, config.vocab_size);
        let mut layers = Vec::with_capacity(config.layers.len());
        for spec in &config.layers {
            let p = format!("model.layers.{}", spec.source_index);
            let ffn = match spec.ffn {
                FfnKind::Dense => {
                    let i = config.dense_intermediate_size;
                    let gate = format!("{p}.mlp.gate_proj");
                    let up = format!("{p}.mlp.up_proj");
                    Ffn::Dense(DenseFfn {
                        gate_up: l.fp8_stacked(&[&gate, &up], i, h)?,
                        down: l.fp8_stacked(&[&format!("{p}.mlp.down_proj")], h, i)?,
                        inter: narrow(i, "dense intermediate size")?,
                    })
                }
                FfnKind::Moe => Ffn::Moe(l.moe(&p)?),
            };
            layers.push(LayerWeights {
                input_norm: l.f32_vec(&format!("{p}.input_layernorm.weight"), h)?,
                post_attention_norm: l
                    .f32_vec(&format!("{p}.post_attention_layernorm.weight"), h)?,
                attention: l.attention(&p, &spec.attention, chunks)?,
                ffn,
            });
        }
        let (embed, lm_head, final_norm) = (
            l.bf16("model.embed_tokens.weight", &[v, h])?,
            l.bf16("lm_head.weight", &[v, h])?,
            l.f32_vec("model.norm.weight", h)?,
        );
        if *l.read.borrow() != expected_tensors(&config) {
            return Err(CudaError::new(
                "the loader read other tensors than its layout check expects",
            ));
        }
        Ok(ModelWeights {
            embed,
            lm_head,
            final_norm,
            layers,
            qkv_chunks: narrow(chunks, "QKV chunks")?,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flash(keep: &[usize]) -> ModelConfig {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../eidola-engine-model/tests/data/flash-mopd.config.json"
        );
        ModelConfig::from_file(std::path::Path::new(path))
            .unwrap()
            .truncated(keep)
            .unwrap()
    }

    /// The layout check accepts exactly the loader's tensors plus the skipped
    /// families and unselected layers, and refuses an extra sink, an unknown
    /// tensor and a missing one.
    #[test]
    fn layouts_are_exact() {
        // Flash: layer 0 global (dense, no sinks), 1 sliding (MoE, sinks), 5 global.
        let c = flash(&[0, 1, 5]);
        let expected = expected_tensors(&c);
        assert!(!expected.contains("model.layers.0.self_attn.attention_sink_bias"));
        assert!(expected.contains("model.layers.1.self_attn.attention_sink_bias"));
        let mut names: Vec<String> = expected.iter().cloned().collect();
        names.extend(
            [
                "visual.blocks.0.attn.sinks",
                "audio_encoder.input_local_transformer.norm.weight",
                "speech_embeddings.3.weight",
                "model.mtp.layers.0.eh_proj.weight",
                "model.layers.2.self_attn.attention_sink_bias",
                "model.layers.47.mlp.experts.255.down_proj.weight",
            ]
            .map(String::from),
        );
        let check = |names: &[String]| check_layout(names.iter().map(String::as_str), &c);
        check(&names).unwrap();
        for extra in [
            "model.layers.0.self_attn.attention_sink_bias",
            "model.layers.5.self_attn.attention_sink_bias",
            "model.layers.1.self_attn.q_norm.weight",
            "model.layers.48.input_layernorm.weight",
            "model.decoder.self_attn.o_proj.weight",
            "model.layers.1.mlp.experts.256.up_proj.weight",
            "rotary_emb.inv_freq",
        ] {
            let mut with = names.clone();
            with.push(extra.into());
            let e = check(&with).expect_err(extra);
            assert!(e.to_string().contains(extra), "{e}");
        }
        for gone in [
            "lm_head.weight",
            "model.layers.1.self_attn.attention_sink_bias",
        ] {
            let without: Vec<String> = names.iter().filter(|n| *n != gone).cloned().collect();
            assert!(check(&without).unwrap_err().to_string().contains("missing"));
        }
    }

    /// The published Flash checkpoint's index (`EIDOLA_MIMO_INDEX`, its
    /// `model.safetensors.index.json`) passes, whole and truncated.
    #[test]
    fn the_published_index_passes() {
        let Some(path) = std::env::var_os("EIDOLA_MIMO_INDEX") else {
            eprintln!("skipping: EIDOLA_MIMO_INDEX not set");
            return;
        };
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let names: Vec<&str> = index["weight_map"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        for keep in [(0..48).collect::<Vec<_>>(), vec![0, 1, 2, 5]] {
            check_layout(names.iter().copied(), &flash(&keep)).unwrap();
        }
    }

    /// A vector shorter (or longer) than the model's width is refused before it
    /// is copied anywhere: the kernels would index it at that width.
    #[test]
    fn short_vectors_are_refused() {
        if !Gpu::available() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let gpu = Gpu::open(0).unwrap();
        let dir = std::env::temp_dir().join(format!("eidola-short-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let header =
            r#"{"model.norm.weight":{"dtype":"BF16","shape":[100],"data_offsets":[0,200]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend(std::iter::repeat_n(0u8, 200));
        std::fs::write(&path, bytes).unwrap();
        let store = WeightSet::open_files(&[path]).unwrap();
        let config_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../eidola-engine-model/tests/data/flash-mopd.config.json"
        );
        let config = ModelConfig::from_file(std::path::Path::new(config_path)).unwrap();
        let l = Loader {
            gpu: &gpu,
            store: &store,
            config: &config,
            read: RefCell::new(BTreeSet::new()),
        };
        let Err(e) = l.f32_vec("model.norm.weight", 4096) else {
            panic!("a short vector loaded");
        };
        assert!(e.to_string().contains("expected [4096]"), "{e}");
        assert!(l.f32_vec("model.norm.weight", 100).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
