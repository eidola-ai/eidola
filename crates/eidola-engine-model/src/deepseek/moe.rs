//! The mixture-of-experts feed-forward: router, routed FP4 experts and the
//! shared FP8 expert (`inference/model.py` `Gate`, `Expert`, `MoE`).

use super::attention::top_k;
use super::config::DsMoeSpec;
use super::linear::{Fp4Linear, Fp8Linear, lin4, lin8};
use super::numeric::{bf16_round, silu32, softplus32};
use crate::tensor::{Matrix, dot};

/// The experts one token goes through, in selection order (best first).
#[derive(Debug, Clone, PartialEq)]
pub struct DsRouting {
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Route one token.
///
/// `score = sqrt(softplus(W·x))` in f32. Selection is the top-k of
/// `score + bias` (higher first, then lower index); the bias steers the
/// choice only. Weights are the unbiased scores of the chosen experts,
/// divided by their sum (in selection order) plus 1e-20 when the config
/// normalises, then multiplied by the route scale.
pub fn route(spec: &DsMoeSpec, router: &Matrix, bias: &[f32], x: &[f32]) -> DsRouting {
    let scores: Vec<f32> = (0..spec.n_routed)
        .map(|e| softplus32(dot(x, router.row(e))).sqrt())
        .collect();
    let biased: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
    let experts = top_k(&biased, 0..spec.n_routed, spec.n_active);
    let mut weights: Vec<f32> = experts.iter().map(|&e| scores[e]).collect();
    if spec.norm_topk_prob && spec.n_active > 1 {
        let sum = weights.iter().fold(0.0f32, |a, w| a + w);
        for w in &mut weights {
            *w /= sum + 1e-20f64 as f32;
        }
    }
    for w in &mut weights {
        *w *= spec.route_scale;
    }
    DsRouting { experts, weights }
}

/// The SwiGLU hidden activation from the two up-projections' BF16 outputs:
/// `silu(min(gate, limit)) · clamp(up, ±limit)` in f32, times the routing
/// weight when there is one, rounded to BF16. The weight multiplies the
/// hidden activation before the down-projection quantises it.
pub fn swiglu_hidden(gate: &[f32], up: &[f32], limit: f32, weight: Option<f32>) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let (g, u) = if limit > 0.0 {
                (g.min(limit), u.clamp(-limit, limit))
            } else {
                (g, u)
            };
            let h = silu32(g) * u;
            bf16_round(match weight {
                Some(w) => w * h,
                None => h,
            })
        })
        .collect()
}

/// One expert's three FP8 linears (the shared expert).
#[derive(Debug, Clone)]
pub struct Fp8Expert {
    pub w1: Fp8Linear,
    pub w2: Fp8Linear,
    pub w3: Fp8Linear,
}

/// One routed expert, decoded from its packed FP4 storage.
pub struct Fp4Expert {
    pub w1: Fp4Linear,
    pub w2: Fp4Linear,
    pub w3: Fp4Linear,
}

/// The shared expert's output for one token (BF16 values).
pub fn shared_expert(e: &Fp8Expert, x: &[f32], limit: f32) -> Vec<f32> {
    let mut gate = vec![0.0; e.w1.rows];
    let mut up = vec![0.0; e.w3.rows];
    lin8(x, &e.w1, &mut gate);
    lin8(x, &e.w3, &mut up);
    let mut out = vec![0.0; e.w2.rows];
    lin8(&swiglu_hidden(&gate, &up, limit, None), &e.w2, &mut out);
    out
}

/// A routed expert's output for one token at routing weight `weight`.
pub fn routed_expert(e: &Fp4Expert, x: &[f32], limit: f32, weight: f32) -> Vec<f32> {
    let mut gate = vec![0.0; e.w1.rows];
    let mut up = vec![0.0; e.w3.rows];
    lin4(x, &e.w1, &mut gate);
    lin4(x, &e.w3, &mut up);
    let mut out = vec![0.0; e.w2.rows];
    lin4(
        &swiglu_hidden(&gate, &up, limit, Some(weight)),
        &e.w2,
        &mut out,
    );
    out
}
