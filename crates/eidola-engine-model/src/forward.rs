//! The dense f32 reference forward pass: the golden oracle every faster
//! executor is diffed against. Clarity over speed; every per-token result is
//! independent of the rest of the batch (see [`crate::tensor`]).

use rayon::prelude::*;

use crate::attention::{apply_rope, attend, rope_cos_sin, visible};
use crate::config::{AttentionSpec, MoeSpec};
use crate::error::{Error, Result};
use crate::tensor::{Matrix, add_assign, dot, linear, rms_norm_rows, sigmoid, silu};
use crate::weights::{AttentionWeights, DenseFfnWeights, FfnWeights, ModelWeights, MtpWeights};

/// Which positions to compute logits for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogitsAt {
    None,
    All,
    Last,
    Positions(Vec<usize>),
}

impl LogitsAt {
    fn resolve(&self, len: usize) -> Result<Vec<usize>> {
        Ok(match self {
            LogitsAt::None => vec![],
            LogitsAt::All => (0..len).collect(),
            LogitsAt::Last => vec![len - 1],
            LogitsAt::Positions(p) => {
                if let Some(&bad) = p.iter().find(|&&i| i >= len) {
                    return Err(Error::Input(format!("logits row {bad} of {len}")));
                }
                p.clone()
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct ForwardOptions {
    pub logits: LogitsAt,
    /// Keep the residual stream after every layer.
    pub capture_layers: bool,
}

impl Default for ForwardOptions {
    fn default() -> Self {
        ForwardOptions {
            logits: LogitsAt::All,
            capture_layers: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ForwardOutput {
    /// Rows of `logits` correspond to these sequence positions.
    pub logit_positions: Vec<usize>,
    /// `[logit_positions.len(), vocab_size]`.
    pub logits: Matrix,
    /// Residual stream after each layer, `[seq, hidden]` each, when captured.
    pub layer_outputs: Vec<Matrix>,
    /// Output of the last layer before the final norm (`[seq, hidden]`).
    pub hidden: Matrix,
    /// `norm(hidden)`, the input to `lm_head`.
    pub hidden_normed: Matrix,
}

/// Router decision for one token.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    /// Selected experts in ascending index order.
    pub experts: Vec<usize>,
    /// Combine weight of each selected expert (same order).
    pub weights: Vec<f32>,
}

/// Sigmoid scoring; selection on `score + correction_bias`; weights from the
/// unbiased scores, normalised over the selection (`Σ` in ascending expert
/// order, `+ 1e-20`), times `routed_scaling_factor`. Ties in the biased score
/// go to the lower expert index.
pub fn route(spec: &MoeSpec, router: &Matrix, bias: &[f32], x: &[f32]) -> Routing {
    let scores: Vec<f32> = (0..spec.num_experts)
        .map(|e| sigmoid(dot(x, router.row(e))))
        .collect();
    let mut order: Vec<usize> = (0..spec.num_experts).collect();
    order.sort_by(|&a, &b| {
        let (ca, cb) = (scores[a] + bias[a], scores[b] + bias[b]);
        cb.total_cmp(&ca).then(a.cmp(&b))
    });
    let mut experts = order[..spec.top_k].to_vec();
    experts.sort_unstable();
    let mut weights: Vec<f32> = experts.iter().map(|&e| scores[e]).collect();
    if spec.top_k > 1 && spec.norm_topk_prob {
        let mut sum = 0.0f32;
        for w in &weights {
            sum += w;
        }
        let denom = sum + 1e-20;
        for w in &mut weights {
            *w /= denom;
        }
    }
    for w in &mut weights {
        *w *= spec.routed_scaling_factor;
    }
    Routing { experts, weights }
}

/// SwiGLU for every row: `down(silu(gate x) · up x)`.
pub fn swiglu(x: &Matrix, gate: &Matrix, up: &Matrix, down: &Matrix) -> Matrix {
    let g = linear(x, gate);
    let u = linear(x, up);
    let mut a = g;
    for (gv, uv) in a.data.iter_mut().zip(&u.data) {
        *gv = silu(*gv) * uv;
    }
    linear(&a, down)
}

pub struct ReferenceModel {
    pub weights: ModelWeights,
}

impl ReferenceModel {
    pub fn new(weights: ModelWeights) -> Self {
        ReferenceModel { weights }
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Matrix> {
        let w = &self.weights;
        let mut x = Matrix::zeros(tokens.len(), w.config.hidden_size);
        for (i, &t) in tokens.iter().enumerate() {
            if t as usize >= w.config.vocab_size {
                return Err(Error::Input(format!(
                    "token {t} outside vocab {}",
                    w.config.vocab_size
                )));
            }
            x.row_mut(i).copy_from_slice(w.embed.row(t as usize));
        }
        Ok(x)
    }

    /// Run the full sequence `tokens` (positions `0..len`) through every
    /// layer.
    pub fn forward(&self, tokens: &[u32], opts: &ForwardOptions) -> Result<ForwardOutput> {
        if tokens.is_empty() {
            return Err(Error::Input("empty token sequence".into()));
        }
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let mut h = self.embed(tokens)?;
        let mut layer_outputs = Vec::new();
        for layer in 0..self.weights.layers.len() {
            self.layer_forward(layer, &mut h, &positions)?;
            if opts.capture_layers {
                layer_outputs.push(h.clone());
            }
        }
        let cfg = &self.weights.config;
        let hidden_normed = rms_norm_rows(&h, &self.weights.final_norm, cfg.rms_norm_eps);
        let logit_positions = opts.logits.resolve(tokens.len())?;
        let logits = self.lm_head(&hidden_normed, &logit_positions);
        Ok(ForwardOutput {
            logit_positions,
            logits,
            layer_outputs,
            hidden: h,
            hidden_normed,
        })
    }

    fn lm_head(&self, normed: &Matrix, rows: &[usize]) -> Matrix {
        if rows.is_empty() {
            return Matrix::zeros(0, self.weights.config.vocab_size);
        }
        linear(&normed.select_rows(rows), &self.weights.lm_head)
    }

    /// One decoder layer, in place on the residual stream `h`.
    pub fn layer_forward(&self, layer: usize, h: &mut Matrix, positions: &[usize]) -> Result<()> {
        let w = &self.weights;
        let lw = &w.layers[layer];
        let spec = &w.config.layers[layer];
        let eps = w.config.rms_norm_eps;

        let x = rms_norm_rows(h, &lw.input_norm, eps);
        let a = self.attention(&spec.attention, &lw.attention, &x, positions)?;
        add_assign(h, &a);

        let x = rms_norm_rows(h, &lw.post_attention_norm, eps);
        let f = match &lw.ffn {
            FfnWeights::Dense(d) => swiglu(&x, &d.gate, &d.up, &d.down),
            FfnWeights::Moe(_) => self.moe(layer, &x)?,
        };
        add_assign(h, &f);
        Ok(())
    }

    /// Attention block (projection, RoPE, attention, `o_proj`) over rows of
    /// `x` at the given strictly increasing positions; row `i` attends to
    /// rows `j ≤ i` visible from it.
    pub fn attention(
        &self,
        spec: &AttentionSpec,
        aw: &AttentionWeights,
        x: &Matrix,
        positions: &[usize],
    ) -> Result<Matrix> {
        if positions.len() != x.rows {
            return Err(Error::Input("one position per row required".into()));
        }
        if positions.windows(2).any(|p| p[1] <= p[0]) {
            return Err(Error::Input("positions must be strictly increasing".into()));
        }
        let t = x.rows;
        let (dq, dv) = (spec.head_dim_qk, spec.head_dim_v);
        let (nq, nkv) = (spec.num_q_heads, spec.num_kv_heads);
        let qkv = linear(x, &aw.qkv);
        let v_scale = if self.weights.value_scale_folded {
            None
        } else {
            self.weights.config.attention_value_scale
        };

        let mut q = Matrix::zeros(t, nq * dq);
        let mut k = Matrix::zeros(t, nkv * dq);
        let mut v = Matrix::zeros(t, nkv * dv);
        for (i, &pos) in positions.iter().enumerate() {
            let row = qkv.row(i);
            q.row_mut(i).copy_from_slice(&row[..nq * dq]);
            k.row_mut(i).copy_from_slice(&row[nq * dq..(nq + nkv) * dq]);
            v.row_mut(i).copy_from_slice(&row[(nq + nkv) * dq..]);
            if let Some(s) = v_scale {
                for x in v.row_mut(i) {
                    *x *= s;
                }
            }
            let (cos, sin) = rope_cos_sin(spec, pos);
            for hq in q.row_mut(i).chunks_exact_mut(dq) {
                apply_rope(hq, &cos, &sin);
            }
            for hk in k.row_mut(i).chunks_exact_mut(dq) {
                apply_rope(hk, &cos, &sin);
            }
        }

        let group = spec.group_size();
        let mut out = Matrix::zeros(t, nq * dv);
        out.data
            .par_chunks_mut(nq * dv)
            .enumerate()
            .for_each(|(i, orow)| {
                let visible_rows: Vec<usize> = (0..=i)
                    .filter(|&j| visible(spec, positions[i], positions[j]))
                    .collect();
                for head in 0..nq {
                    let g = head / group;
                    let keys: Vec<&[f32]> = visible_rows
                        .iter()
                        .map(|&j| &k.row(j)[g * dq..(g + 1) * dq])
                        .collect();
                    let values: Vec<&[f32]> = visible_rows
                        .iter()
                        .map(|&j| &v.row(j)[g * dv..(g + 1) * dv])
                        .collect();
                    let sink = aw.sinks.as_ref().map(|s| s[head]);
                    attend(
                        spec,
                        &q.row(i)[head * dq..(head + 1) * dq],
                        &keys,
                        &values,
                        sink,
                        &mut orow[head * dv..(head + 1) * dv],
                    );
                }
            });
        Ok(linear(&out, &aw.o_proj))
    }

    /// Routed-expert block for every row of `x`. Each selected expert's output
    /// is scaled by its weight and accumulated in ascending expert order.
    pub fn moe(&self, layer: usize, x: &Matrix) -> Result<Matrix> {
        let w = &self.weights;
        let FfnWeights::Moe(mw) = &w.layers[layer].ffn else {
            return Err(Error::Input(format!("layer {layer} is not MoE")));
        };
        let spec = w.config.moe.as_ref().expect("MoE layer implies MoE config");
        let routes: Vec<Routing> = (0..x.rows)
            .into_par_iter()
            .map(|i| route(spec, &mw.router, &mw.correction_bias, x.row(i)))
            .collect();

        // Rows per expert, so each expert is dequantised once per call.
        let mut by_expert: Vec<Vec<usize>> = vec![Vec::new(); spec.num_experts];
        for (i, r) in routes.iter().enumerate() {
            for &e in &r.experts {
                by_expert[e].push(i);
            }
        }
        let outputs: Vec<Option<Matrix>> = by_expert
            .par_iter()
            .enumerate()
            .map(|(e, rows)| -> Result<Option<Matrix>> {
                if rows.is_empty() {
                    return Ok(None);
                }
                let ew = w.expert(layer, e)?;
                Ok(Some(swiglu(
                    &x.select_rows(rows),
                    &ew.gate,
                    &ew.up,
                    &ew.down,
                )))
            })
            .collect::<Result<_>>()?;

        let mut out = Matrix::zeros(x.rows, x.cols);
        let mut cursor = vec![0usize; spec.num_experts];
        for (i, r) in routes.iter().enumerate() {
            let orow = out.row_mut(i);
            for (&e, &wt) in r.experts.iter().zip(&r.weights) {
                let y = outputs[e].as_ref().expect("routed expert ran");
                let yrow = y.row(cursor[e]);
                cursor[e] += 1;
                for (o, v) in orow.iter_mut().zip(yrow) {
                    *o += v * wt;
                }
            }
        }
        Ok(out)
    }

    /// Router decisions of MoE layer `layer` for every row of `x` (the
    /// post-attention-normed residual), for diffing executors.
    pub fn routing(&self, layer: usize, x: &Matrix) -> Result<Vec<Routing>> {
        let w = &self.weights;
        let FfnWeights::Moe(mw) = &w.layers[layer].ffn else {
            return Err(Error::Input(format!("layer {layer} is not MoE")));
        };
        let spec = w.config.moe.as_ref().expect("MoE layer implies MoE config");
        Ok((0..x.rows)
            .map(|i| route(spec, &mw.router, &mw.correction_bias, x.row(i)))
            .collect())
    }

    pub fn dense_ffn(&self, d: &DenseFfnWeights, x: &Matrix) -> Matrix {
        swiglu(x, &d.gate, &d.up, &d.down)
    }
}

/// Output of one multi-token-prediction layer.
#[derive(Debug, Clone)]
pub struct MtpOutput {
    pub logit_rows: Vec<usize>,
    pub logits: Matrix,
    /// Layer output before `final_layernorm`.
    pub hidden: Matrix,
    /// `final_layernorm(hidden)`, the input to the shared `lm_head`.
    pub hidden_normed: Matrix,
}

impl ReferenceModel {
    /// MTP layer `k` over a sequence of rows. Row `i` combines the embedding
    /// of `tokens[i]` with `prev_hidden` row `i`:
    /// `eh_proj([enorm(embed(tokens[i])) ‖ hnorm(prev_hidden[i])])`, then a
    /// sliding-window attention block and a dense SwiGLU block (pre-norm,
    /// residual), then `final_layernorm` and the shared `lm_head`. Rows
    /// attend causally to earlier rows by `positions`.
    ///
    /// How rows line up with the main model (which token, which hidden
    /// state, which position) is the caller's choice and is not encoded here.
    pub fn mtp_forward(
        &self,
        k: usize,
        tokens: &[u32],
        prev_hidden: &Matrix,
        positions: &[usize],
        logits: &LogitsAt,
    ) -> Result<MtpOutput> {
        let w = &self.weights;
        let mw: &MtpWeights = w
            .mtp
            .get(k)
            .ok_or_else(|| Error::Input(format!("MTP layer {k} is not loaded")))?;
        let cfg = &w.config;
        let hsz = cfg.hidden_size;
        if prev_hidden.rows != tokens.len() || prev_hidden.cols != hsz {
            return Err(Error::Input("prev_hidden must be [tokens, hidden]".into()));
        }
        let eps = cfg.rms_norm_eps;
        let e = rms_norm_rows(&self.embed(tokens)?, &mw.enorm, eps);
        let hp = rms_norm_rows(prev_hidden, &mw.hnorm, eps);
        let mut cat = Matrix::zeros(tokens.len(), 2 * hsz);
        for i in 0..tokens.len() {
            let r = cat.row_mut(i);
            r[..hsz].copy_from_slice(e.row(i));
            r[hsz..].copy_from_slice(hp.row(i));
        }
        let mut h = linear(&cat, &mw.eh_proj);

        let x = rms_norm_rows(&h, &mw.input_norm, eps);
        let a = self.attention(&cfg.mtp.attention, &mw.attention, &x, positions)?;
        add_assign(&mut h, &a);
        let x = rms_norm_rows(&h, &mw.post_attention_norm, eps);
        let f = self.dense_ffn(&mw.ffn, &x);
        add_assign(&mut h, &f);

        let hidden_normed = rms_norm_rows(&h, &mw.final_norm, eps);
        let logit_rows = logits.resolve(tokens.len())?;
        let logits = self.lm_head(&hidden_normed, &logit_rows);
        Ok(MtpOutput {
            logit_rows,
            logits,
            hidden: h,
            hidden_normed,
        })
    }
}
