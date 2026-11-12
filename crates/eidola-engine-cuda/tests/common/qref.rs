//! The f32 reference forward with the CUDA executor's quantization points
//! emulated (fake quantization: quantize, then dequantize, in f32), to
//! attribute the executor's divergence from the f32 reference: what this
//! emulation already loses against the reference is the cost of the
//! quantization recipe; what remains between it and the GPU is accumulation
//! order, BF16 attention probabilities and the like.
//!
//! Emulated, as the executor does them:
//! - FP8 e4m3 activations with f32 per-128 scales (`amax / 448`) into every
//!   dense FP8 GEMM (QKV, dense gate/up, dense down);
//! - FP8 e4m3 activations with power-of-two per-128 scales into the expert
//!   GEMMs (gate/up input and the SwiGLU product into down);
//! - BF16 rounding of every GEMM output the executor keeps in BF16 (QKV,
//!   dense and expert gate/up and down), of rotated Q and K, of the attention
//!   output, and of the final normed hidden state fed to `lm_head`.
//!
//! Not emulated: BF16 attention probabilities inside the attention kernel,
//! f32 accumulation order in the GEMMs and norms.
#![allow(dead_code)]

use eidola_engine_model::ReferenceModel;
use eidola_engine_model::attention::{apply_rope, attend, rope_cos_sin, visible};
use eidola_engine_model::forward::route;
use eidola_engine_model::tensor::{Matrix, add_assign, linear, rms_norm_rows, silu};
use eidola_engine_model::weights::FfnWeights;

use eidola_engine_cuda::bf16;

/// Round to the nearest e4m3 value, ties to even, saturating at ±448.
pub fn e4m3(x: f32) -> f32 {
    let a = x.abs();
    if a == 0.0 || a.is_nan() {
        return x;
    }
    // A finite nonzero f32's binary exponent lies in [-149, 127].
    #[expect(clippy::cast_possible_truncation, reason = "an f32 exponent fits i32")]
    let e = a.log2().floor() as i32;
    let step = if e < -6 {
        2f32.powi(-9)
    } else {
        2f32.powi(e - 3)
    };
    let q = ((a / step).round_ties_even() * step).min(448.0);
    q.copysign(x)
}

fn bf(x: &mut [f32]) {
    for v in x {
        *v = bf16::to_f32(bf16::from_f32(*v));
    }
}

/// Per row, per 128 of K: scale `amax / 448` (1 for zeros), `q = e4m3(x / s)`.
fn fake_fp8_f32(x: &Matrix) -> Matrix {
    let mut out = x.clone();
    for r in 0..x.rows {
        for g in out.row_mut(r).chunks_mut(128) {
            let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 448.0 } else { 1.0 };
            for v in g {
                *v = e4m3(*v / s) * s;
            }
        }
    }
    out
}

/// Per row, per 128 of K: scale `2^ceil(log2(amax / 448))`.
fn fake_fp8_ue8m0(x: &Matrix) -> Matrix {
    let mut out = x.clone();
    for r in 0..x.rows {
        for g in out.row_mut(r).chunks_mut(128) {
            let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
            let e = (amax.max(1e-10) / 448.0).log2().ceil().clamp(-127.0, 127.0);
            #[expect(clippy::cast_possible_truncation, reason = "clamped to [-127, 127]")]
            let s = 2f32.powi(e as i32);
            for v in g {
                *v = e4m3(*v / s) * s;
            }
        }
    }
    out
}

fn bf_linear(x: &Matrix, w: &Matrix) -> Matrix {
    let mut y = linear(x, w);
    bf(&mut y.data);
    y
}

fn swiglu_quant(
    x: &Matrix,
    gate: &Matrix,
    up: &Matrix,
    down: &Matrix,
    quant: fn(&Matrix) -> Matrix,
) -> Matrix {
    let xq = quant(x);
    let g = bf_linear(&xq, gate);
    let u = bf_linear(&xq, up);
    let mut a = g;
    for (gv, uv) in a.data.iter_mut().zip(&u.data) {
        *gv = silu(*gv) * uv;
    }
    bf_linear(&quant(&a), down)
}

/// Residual stream after every layer and the logits at every position.
pub fn forward(m: &ReferenceModel, tokens: &[u32]) -> (Vec<Matrix>, Matrix) {
    let w = &m.weights;
    let cfg = &w.config;
    let eps = cfg.rms_norm_eps;
    let mut h = Matrix::zeros(tokens.len(), cfg.hidden_size);
    for (i, &t) in tokens.iter().enumerate() {
        h.row_mut(i).copy_from_slice(w.embed.row(t as usize));
    }
    let mut layers = Vec::new();
    for (l, lw) in w.layers.iter().enumerate() {
        let spec = &cfg.layers[l].attention;
        let x = rms_norm_rows(&h, &lw.input_norm, eps);
        let qkv = bf_linear(&fake_fp8_f32(&x), &lw.attention.qkv);
        let (dq, dv, nq, nkv) = (
            spec.head_dim_qk,
            spec.head_dim_v,
            spec.num_q_heads,
            spec.num_kv_heads,
        );
        let t = tokens.len();
        let mut q = Matrix::zeros(t, nq * dq);
        let mut k = Matrix::zeros(t, nkv * dq);
        let mut v = Matrix::zeros(t, nkv * dv);
        for i in 0..t {
            let row = qkv.row(i);
            q.row_mut(i).copy_from_slice(&row[..nq * dq]);
            k.row_mut(i).copy_from_slice(&row[nq * dq..(nq + nkv) * dq]);
            v.row_mut(i).copy_from_slice(&row[(nq + nkv) * dq..]);
            let (cos, sin) = rope_cos_sin(spec, i);
            for hq in q.row_mut(i).chunks_exact_mut(dq) {
                apply_rope(hq, &cos, &sin);
            }
            for hk in k.row_mut(i).chunks_exact_mut(dq) {
                apply_rope(hk, &cos, &sin);
            }
        }
        bf(&mut q.data);
        bf(&mut k.data);
        let mut out = Matrix::zeros(t, nq * dv);
        let group = spec.group_size();
        for i in 0..t {
            let rows: Vec<usize> = (0..=i).filter(|&j| visible(spec, i, j)).collect();
            for head in 0..nq {
                let g = head / group;
                let keys: Vec<&[f32]> = rows
                    .iter()
                    .map(|&j| &k.row(j)[g * dq..(g + 1) * dq])
                    .collect();
                let values: Vec<&[f32]> = rows
                    .iter()
                    .map(|&j| &v.row(j)[g * dv..(g + 1) * dv])
                    .collect();
                let sink = lw.attention.sinks.as_ref().map(|s| s[head]);
                let o = &mut out.row_mut(i)[head * dv..(head + 1) * dv];
                attend(
                    spec,
                    &q.row(i)[head * dq..(head + 1) * dq],
                    &keys,
                    &values,
                    sink,
                    o,
                );
            }
        }
        bf(&mut out.data);
        add_assign(&mut h, &linear(&out, &lw.attention.o_proj));

        let x = rms_norm_rows(&h, &lw.post_attention_norm, eps);
        let f = match &lw.ffn {
            FfnWeights::Dense(d) => swiglu_quant(&x, &d.gate, &d.up, &d.down, fake_fp8_f32),
            FfnWeights::Moe(mw) => {
                let ms = cfg.moe.as_ref().unwrap();
                let routes: Vec<_> = (0..t)
                    .map(|i| route(ms, &mw.router, &mw.correction_bias, x.row(i)))
                    .collect();
                let mut by_expert: Vec<Vec<usize>> = vec![Vec::new(); ms.num_experts];
                for (i, r) in routes.iter().enumerate() {
                    for &e in &r.experts {
                        by_expert[e].push(i);
                    }
                }
                let ys: Vec<Option<Matrix>> = by_expert
                    .iter()
                    .enumerate()
                    .map(|(e, rows)| {
                        (!rows.is_empty()).then(|| {
                            let ew = w.expert(l, e).unwrap();
                            swiglu_quant(
                                &x.select_rows(rows),
                                &ew.gate,
                                &ew.up,
                                &ew.down,
                                fake_fp8_ue8m0,
                            )
                        })
                    })
                    .collect();
                let mut out = Matrix::zeros(t, x.cols);
                let mut cursor = vec![0usize; ms.num_experts];
                for (i, r) in routes.iter().enumerate() {
                    let orow = out.row_mut(i);
                    for (&e, &wt) in r.experts.iter().zip(&r.weights) {
                        let yrow = ys[e].as_ref().unwrap().row(cursor[e]);
                        cursor[e] += 1;
                        for (o, v) in orow.iter_mut().zip(yrow) {
                            *o += v * wt;
                        }
                    }
                }
                out
            }
        };
        add_assign(&mut h, &f);
        layers.push(h.clone());
    }
    let mut normed = rms_norm_rows(&h, &w.final_norm, eps);
    bf(&mut normed.data);
    (layers, linear(&normed, &w.lm_head))
}
