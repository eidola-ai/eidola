//! The Qwen3.8 reference forward.
//!
//! The residual stream is BF16. Each layer is
//!
//! ```text
//! h ← bf16(h + mixer(norm_in(h)));   h ← bf16(h + mlp(norm_post(h)))
//! ```
//!
//! with zero-centred RMSNorms (`bf16(x · rsqrt(mean(x²) + eps) · (1 + w))`, one
//! rounding), the mixer a Gated DeltaNet ([`super::deltanet`]) or gated full
//! attention ([`super::attention`]), and the MLP
//! `down(bf16(bf16(silu(gate(x))) · up(x)))`. After the last layer the final
//! norm, then the untied head, its logits rounded to BF16.
//!
//! [`QwenModel::layer_forward`] runs one layer over a contiguous range of
//! positions from the state the positions before it left, so the full forward,
//! any chunking of it and a resume from a saved [`SequenceState`] agree bit
//! for bit.

use rayon::prelude::*;

use super::attention::{apply_rope, attend, inv_freq, rope_cos_sin};
use super::config::{LayerKind, QwenConfig};
use super::deltanet::{CoreWeights, DeltaNetState, token_core};
use super::linear::{lin8_rows, linear_bf16_dense, linear_bf16_rows};
use super::numeric::{bf16_round, rsqrt32, sigmoid32, silu32};
use super::weights::{
    AttentionWeights, LayerWeights, MixerWeights, MlpWeights, QwenWeights, ZcNorm,
};
use crate::error::{Error, Result};
use crate::tensor::{Matrix, dot};

/// The zero-centred RMSNorm of one row: `bf16((x · r) · (1 + w))` with
/// `r = rsqrt(Σx² / n + eps)`, the sum in the pinned order.
pub fn rms_norm_zc(x: &[f32], one_plus_w: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), one_plus_w.len());
    let r = rsqrt32(dot(x, x) / x.len() as f32 + eps);
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(one_plus_w) {
        *o = bf16_round((v * r) * w);
    }
}

pub fn rms_norm_zc_rows(x: &Matrix, norm: &ZcNorm, eps: f32) -> Matrix {
    let mut out = Matrix::zeros(x.rows, x.cols);
    out.data
        .par_chunks_mut(x.cols.max(1))
        .enumerate()
        .for_each(|(r, o)| rms_norm_zc(x.row(r), &norm.0, eps, o));
    out
}

/// `bf16(a + b)` element-wise: the residual update.
pub fn residual_add(a: &mut Matrix, b: &Matrix) {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols));
    for (x, y) in a.data.iter_mut().zip(&b.data) {
        *x = bf16_round(*x + *y);
    }
}

/// KV of one attention layer: per position, `kv_heads · head_dim` keys (after
/// the norm and RoPE) and values, BF16 values.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct KvCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

impl KvCache {
    pub fn len(&self, width: usize) -> usize {
        self.keys.len() / width
    }
}

/// One layer's state between positions.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerState {
    DeltaNet(DeltaNetState),
    Attention(KvCache),
}

/// Everything a sequence carries from one position to the next. Cloning it is
/// a snapshot: running on from a clone equals running on from the original.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceState {
    pub layers: Vec<LayerState>,
    /// The next position to run.
    pub position: usize,
}

/// Which positions of a forward call get logits.
#[derive(Clone, Debug, Default)]
pub enum LogitRows {
    #[default]
    None,
    Last,
    All,
    /// Offsets within the call.
    At(Vec<usize>),
}

#[derive(Clone, Debug, Default)]
pub struct QwenForwardOptions {
    pub logits: LogitRows,
    /// Keep every layer's output residual stream.
    pub keep_streams: bool,
}

pub struct QwenForwardOutput {
    /// Offsets within the call of the rows of `logits`.
    pub logit_rows: Vec<usize>,
    /// `[logit_rows, vocab]`, BF16 values.
    pub logits: Matrix,
    /// After the final norm, every position of the call: what the drafter
    /// reads.
    pub normed: Matrix,
    /// Per layer, the residual stream it leaves, when asked for.
    pub streams: Vec<Matrix>,
}

pub struct MtpOutput {
    /// The layer's output residual stream, before the drafter's norm.
    pub hidden: Matrix,
    /// After the drafter's norm: the input the next draft step takes.
    pub normed: Matrix,
    pub logit_rows: Vec<usize>,
    pub logits: Matrix,
}

pub struct QwenModel {
    pub weights: QwenWeights,
    inv_freq: Vec<f32>,
}

impl QwenModel {
    pub fn new(weights: QwenWeights) -> Self {
        let inv_freq = inv_freq(&weights.config.attention);
        QwenModel { weights, inv_freq }
    }

    pub fn config(&self) -> &QwenConfig {
        &self.weights.config
    }

    pub fn new_state(&self) -> SequenceState {
        let c = self.config();
        SequenceState {
            layers: c
                .layers
                .iter()
                .map(|l| match l.kind {
                    LayerKind::DeltaNet => LayerState::DeltaNet(DeltaNetState::new(&c.deltanet)),
                    LayerKind::Attention => LayerState::Attention(KvCache::default()),
                })
                .collect(),
            position: 0,
        }
    }

    fn check_tokens(&self, tokens: &[u32]) -> Result<()> {
        let c = self.config();
        for &t in tokens {
            if t as usize >= c.vocab_size {
                return Err(Error::Input(format!("token {t} is outside the vocabulary")));
            }
            if c.vision_token_ids.contains(&t) {
                return Err(Error::Input(format!(
                    "token {t} is a vision placeholder; vision is not served"
                )));
            }
        }
        Ok(())
    }

    /// The embedding rows of `tokens`.
    pub fn embed(&self, tokens: &[u32]) -> Result<Matrix> {
        self.check_tokens(tokens)?;
        let h = self.config().hidden_size;
        let mut out = Matrix::zeros(tokens.len(), h);
        for (r, &t) in tokens.iter().enumerate() {
            self.weights.embed.row_into(t as usize, out.row_mut(r));
        }
        Ok(out)
    }

    /// The RoPE tables at `position`.
    pub fn rope(&self, position: usize) -> (Vec<f32>, Vec<f32>) {
        rope_cos_sin(&self.inv_freq, position)
    }

    /// The gated attention mixer over normed rows at positions
    /// `start..start + rows`. `cache` holds exactly the positions before.
    pub fn attention_mix(
        &self,
        w: &AttentionWeights,
        cache: &mut KvCache,
        x: &Matrix,
        start: usize,
    ) -> Result<Matrix> {
        let a = &self.config().attention;
        let eps = self.config().rms_norm_eps;
        let (d, nq, nkv) = (a.head_dim, a.num_q_heads, a.num_kv_heads);
        let width = nkv * d;
        if cache.len(width) != start {
            return Err(Error::Input(format!(
                "attention cache holds {} positions, the call starts at {start}",
                cache.len(width)
            )));
        }
        let qg = lin8_rows(x, &w.q_proj);
        let k = lin8_rows(x, &w.k_proj);
        let v = lin8_rows(x, &w.v_proj);
        let t = x.rows;
        let mut q = Matrix::zeros(t, nq * d);
        let mut gate = Matrix::zeros(t, nq * d);
        for r in 0..t {
            let (cos, sin) = self.rope(start + r);
            for h in 0..nq {
                let src = &qg.row(r)[2 * h * d..(2 * h + 2) * d];
                let qh = &mut q.row_mut(r)[h * d..(h + 1) * d];
                rms_norm_zc(&src[..d], &w.q_norm.0, eps, qh);
                apply_rope(qh, &cos, &sin);
                gate.row_mut(r)[h * d..(h + 1) * d].copy_from_slice(&src[d..]);
            }
            let mut kr = vec![0.0f32; width];
            for g in 0..nkv {
                let kh = &mut kr[g * d..(g + 1) * d];
                rms_norm_zc(&k.row(r)[g * d..(g + 1) * d], &w.k_norm.0, eps, kh);
                apply_rope(kh, &cos, &sin);
            }
            cache.keys.extend_from_slice(&kr);
            cache.values.extend_from_slice(v.row(r));
        }
        let group = a.group();
        let scale = a.softmax_scale;
        let mut y = Matrix::zeros(t, nq * d);
        y.data.par_chunks_mut(d).enumerate().for_each(|(i, out)| {
            let (r, h) = (i / nq, i % nq);
            let g = h / group;
            let visible = start + r + 1;
            let keys: Vec<&[f32]> = (0..visible)
                .map(|p| &cache.keys[p * width + g * d..p * width + (g + 1) * d])
                .collect();
            let values: Vec<&[f32]> = (0..visible)
                .map(|p| &cache.values[p * width + g * d..p * width + (g + 1) * d])
                .collect();
            attend(&q.row(r)[h * d..(h + 1) * d], &keys, &values, scale, out);
            let gr = &gate.row(r)[h * d..(h + 1) * d];
            for (o, &gv) in out.iter_mut().zip(gr) {
                *o = bf16_round(*o * bf16_round(sigmoid32(gv)));
            }
        });
        Ok(lin8_rows(&y, &w.o_proj))
    }

    /// The Gated DeltaNet mixer over normed rows: the projections for every
    /// row at once, then [`token_core`] for each row in order from `state`,
    /// then the output projection.
    pub fn deltanet_mix(
        &self,
        w: &super::weights::DeltaNetWeights,
        state: &mut DeltaNetState,
        x: &Matrix,
    ) -> Matrix {
        let spec = &self.config().deltanet;
        let eps = self.config().rms_norm_eps;
        let mixed = lin8_rows(x, &w.in_proj_qkv);
        let z = lin8_rows(x, &w.in_proj_z);
        let b = linear_bf16_dense(x, &w.in_proj_b);
        let a = linear_bf16_dense(x, &w.in_proj_a);
        let core = CoreWeights {
            conv: &w.conv,
            a_log: &w.a_log,
            dt_bias: &w.dt_bias,
            norm: &w.norm,
        };
        let mut n = Matrix::zeros(x.rows, spec.value_dim());
        for r in 0..x.rows {
            let o = token_core(
                spec,
                state,
                mixed.row(r),
                z.row(r),
                a.row(r),
                b.row(r),
                &core,
                eps,
            );
            n.row_mut(r).copy_from_slice(&o);
        }
        lin8_rows(&n, &w.out_proj)
    }

    pub fn mlp(&self, w: &MlpWeights, x: &Matrix) -> Matrix {
        let g = lin8_rows(x, &w.gate);
        let mut u = lin8_rows(x, &w.up);
        for (uv, &gv) in u.data.iter_mut().zip(&g.data) {
            *uv = bf16_round(bf16_round(silu32(gv)) * *uv);
        }
        lin8_rows(&u, &w.down)
    }

    fn run_layer(
        &self,
        w: &LayerWeights,
        state: &mut LayerState,
        h: &Matrix,
        start: usize,
    ) -> Result<Matrix> {
        let eps = self.config().rms_norm_eps;
        let x = rms_norm_zc_rows(h, &w.input_norm, eps);
        let mix = match (&w.mixer, state) {
            (MixerWeights::DeltaNet(dw), LayerState::DeltaNet(s)) => self.deltanet_mix(dw, s, &x),
            (MixerWeights::Attention(aw), LayerState::Attention(c)) => {
                self.attention_mix(aw, c, &x, start)?
            }
            _ => return Err(Error::Input("layer state of the wrong kind".into())),
        };
        let mut h = h.clone();
        residual_add(&mut h, &mix);
        let x = rms_norm_zc_rows(&h, &w.post_norm, eps);
        let m = self.mlp(&w.mlp, &x);
        residual_add(&mut h, &m);
        Ok(h)
    }

    /// One decoder layer over the residual rows `h` at positions
    /// `start..start + rows`, from the state the positions before left.
    pub fn layer_forward(
        &self,
        layer: usize,
        state: &mut LayerState,
        h: &Matrix,
        start: usize,
    ) -> Result<Matrix> {
        self.run_layer(&self.weights.layers[layer], state, h, start)
    }

    pub fn final_norm(&self, h: &Matrix) -> Matrix {
        rms_norm_zc_rows(h, &self.weights.final_norm, self.config().rms_norm_eps)
    }

    /// Logits of normed rows: f32 dot products in the pinned order, rounded
    /// to BF16.
    pub fn head(&self, normed: &Matrix) -> Matrix {
        linear_bf16_rows(normed, &self.weights.head)
    }

    fn select(rows: &LogitRows, n: usize) -> Result<Vec<usize>> {
        let v = match rows {
            LogitRows::None => vec![],
            LogitRows::Last => n.checked_sub(1).into_iter().collect(),
            LogitRows::All => (0..n).collect(),
            LogitRows::At(v) => v.clone(),
        };
        if let Some(bad) = v.iter().find(|&&r| r >= n) {
            return Err(Error::Input(format!("logit row {bad} of a {n}-token call")));
        }
        Ok(v)
    }

    /// Run `tokens` at the state's next positions.
    pub fn forward(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        opts: &QwenForwardOptions,
    ) -> Result<QwenForwardOutput> {
        if state.layers.len() != self.weights.layers.len() {
            return Err(Error::Input("state made for another model".into()));
        }
        let logit_rows = Self::select(&opts.logits, tokens.len())?;
        let mut h = self.embed(tokens)?;
        let start = state.position;
        let mut streams = Vec::new();
        for (l, s) in state.layers.iter_mut().enumerate() {
            h = self.layer_forward(l, s, &h, start)?;
            if opts.keep_streams {
                streams.push(h.clone());
            }
        }
        state.position += tokens.len();
        let normed = self.final_norm(&h);
        let logits = self.head(&normed.select_rows(&logit_rows));
        Ok(QwenForwardOutput {
            logit_rows,
            logits,
            normed,
            streams,
        })
    }

    /// MTP layer `depth` over rows at positions `start..start + rows`: row `r`
    /// combines the embedding of `tokens[r]` with `hidden.row(r)` (whatever the
    /// caller chooses to feed: the target's normed state at that position, or
    /// a previous draft step's normed output). `cache` is the layer's own KV
    /// and holds exactly the positions before `start`.
    pub fn mtp_forward(
        &self,
        depth: usize,
        cache: &mut KvCache,
        start: usize,
        tokens: &[u32],
        hidden: &Matrix,
        logits: &LogitRows,
    ) -> Result<MtpOutput> {
        let w = self
            .weights
            .mtp
            .get(depth)
            .ok_or_else(|| Error::Input(format!("no MTP layer {depth}")))?;
        let hsz = self.config().hidden_size;
        if hidden.rows != tokens.len() || hidden.cols != hsz {
            return Err(Error::Input(
                "MTP hidden rows do not match its tokens".into(),
            ));
        }
        let eps = self.config().rms_norm_eps;
        let logit_rows = Self::select(logits, tokens.len())?;
        let e = rms_norm_zc_rows(&self.embed(tokens)?, &w.pre_fc_norm_embedding, eps);
        let hn = rms_norm_zc_rows(hidden, &w.pre_fc_norm_hidden, eps);
        let mut cat = Matrix::zeros(tokens.len(), 2 * hsz);
        for r in 0..tokens.len() {
            cat.row_mut(r)[..hsz].copy_from_slice(e.row(r));
            cat.row_mut(r)[hsz..].copy_from_slice(hn.row(r));
        }
        let u = linear_bf16_rows(&cat, &w.fc);
        let mut state = LayerState::Attention(std::mem::take(cache));
        let out = self.run_layer(&w.layer, &mut state, &u, start);
        if let LayerState::Attention(c) = state {
            *cache = c;
        }
        let hidden = out?;
        let normed = rms_norm_zc_rows(&hidden, &w.norm, eps);
        let logits = self.head(&normed.select_rows(&logit_rows));
        Ok(MtpOutput {
            hidden,
            normed,
            logit_rows,
            logits,
        })
    }
}
