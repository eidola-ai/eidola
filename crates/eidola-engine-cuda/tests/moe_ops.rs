//! The expert path's router (every form), gather, SwiGLU and combine kernels
//! against their single-block reference forms (`engine_ops_reference.cu`, the
//! numerics they must keep) bit for bit, and the router against the model
//! crate's routing, on every image this device runs, from one token to a full
//! prefill step.

mod common;

use common::{Lcg, setup};
use eidola_engine_cuda::bf16;
use eidola_engine_cuda::engine_ops::{
    COMBINE_WIDTH, EngineOps, ROUTER_CLUSTER, ROUTER_SCORES_SMEM, ROUTER_SCORES_THREADS,
    ROUTER_SELECT_WARPS, ROUTER_THREADS, ROUTER_TILED_THREADS, RouterForm, expert_placement,
    psum_rows, router_scores_len,
};
use eidola_engine_cuda::launch::dptr;
use eidola_engine_cuda::{Gpu, ImageArch, Kernel};
use eidola_engine_model::Matrix;
use eidola_engine_model::config::MoeSpec;
use eidola_engine_model::forward::route;

const HIDDEN: usize = 4096;
const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const INTER: usize = 2048;
/// Token counts: decode rows, decode batches (64 and 128 tokens, and 129, one
/// past a block of rows), 513 (which leaves the tiled router a one-token last
/// tile), and a full prefill step.
const TOKENS: [usize; 9] = [1, 2, 7, 64, 128, 129, 513, 2048, 8192];
/// Every router form, whatever the token count.
const FORMS: [RouterForm; 3] = [RouterForm::PerToken, RouterForm::Tiled, RouterForm::Split];

fn u32_of(x: usize) -> u32 {
    u32::try_from(x).unwrap()
}

/// The reference kernels, launched as they were: one block per token (router,
/// combine) or per layout row (gather, SwiGLU).
struct Reference {
    router: Kernel,
    swiglu: Kernel,
    gather: Kernel,
    combine: Kernel,
}

impl Reference {
    fn load(su: &common::Setup, arch: ImageArch) -> Reference {
        let m = su.module("engine_ops_reference", arch);
        Reference {
            router: m.kernel("eidola_reference_router_topk").unwrap(),
            swiglu: m.kernel("eidola_reference_swiglu_quant_fp8_ue8m0").unwrap(),
            gather: m.kernel("eidola_reference_gather_quant_ue8m0").unwrap(),
            combine: m.kernel("eidola_reference_moe_combine").unwrap(),
        }
    }
}

struct RouterCase {
    x: Vec<f32>,
    w: Vec<u16>,
    bias: Vec<f32>,
}

/// Random rows with ties built in: experts 2, 3, 4, 130 and 255 share one
/// weight row and one bias (equal choices for every token), every fifth token
/// is zero (every score 0.5, so the choice is the bias, which takes only 16
/// values), and every seventh token repeats the previous one.
fn router_case(rng: &mut Lcg, tokens: usize) -> RouterCase {
    let mut w: Vec<u16> = (0..EXPERTS * HIDDEN)
        .map(|_| bf16::from_f32(rng.f32() * 0.03))
        .collect();
    let shared = w[2 * HIDDEN..3 * HIDDEN].to_vec();
    for e in [3, 4, 130, 255] {
        w[e * HIDDEN..(e + 1) * HIDDEN].copy_from_slice(&shared);
    }
    let mut bias: Vec<f32> = (0..EXPERTS)
        .map(|e| ((e * 37) % 16) as f32 * 1e-3)
        .collect();
    for e in [3, 4, 130, 255] {
        bias[e] = bias[2];
    }
    let mut x = vec![0f32; tokens * HIDDEN];
    for t in 0..tokens {
        let row = t * HIDDEN;
        if t % 5 == 0 {
            continue;
        }
        if t % 7 == 0 {
            x.copy_within(row - HIDDEN..row, row);
            continue;
        }
        for v in &mut x[row..row + HIDDEN] {
            *v = rng.f32();
        }
    }
    RouterCase { x, w, bias }
}

/// Runs the executor's router in `form` and the reference over one case;
/// returns (ids, weight bits) of each.
#[allow(clippy::type_complexity)]
fn run_router(
    gpu: &Gpu,
    ops: &EngineOps,
    form: RouterForm,
    reference: &Reference,
    case: &RouterCase,
    tokens: usize,
) -> ((Vec<i32>, Vec<u32>), (Vec<i32>, Vec<u32>)) {
    run_router_experts(gpu, ops, form, reference, case, tokens, EXPERTS)
}

/// [`run_router`] over the first `experts` of the case's experts. The split
/// form's scratch starts as NaN, so a score it selects over without writing
/// shows.
#[allow(clippy::type_complexity)]
fn run_router_experts(
    gpu: &Gpu,
    ops: &EngineOps,
    form: RouterForm,
    reference: &Reference,
    case: &RouterCase,
    tokens: usize,
    experts: usize,
) -> ((Vec<i32>, Vec<u32>), (Vec<i32>, Vec<u32>)) {
    let s = gpu.stream();
    let x = s.clone_htod(&case.x).unwrap();
    let w = s.clone_htod(&case.w).unwrap();
    let bias = s.clone_htod(&case.bias).unwrap();
    let scores = s
        .clone_htod(&vec![f32::NAN; router_scores_len(tokens, experts).unwrap()])
        .unwrap();
    let mut out = Vec::new();
    for new in [true, false] {
        let ids = s.alloc_zeros::<i32>(tokens * TOP_K).unwrap();
        let wts = s.alloc_zeros::<f32>(tokens * TOP_K).unwrap();
        unsafe {
            if new {
                ops.router_topk_form(
                    gpu,
                    form,
                    dptr(&ids, s),
                    dptr(&wts, s),
                    dptr(&scores, s),
                    dptr(&x, s),
                    dptr(&w, s),
                    dptr(&bias, s),
                    u32_of(tokens),
                    u32_of(HIDDEN),
                    u32_of(experts),
                    u32_of(TOP_K),
                    1.0f32,
                )
                .unwrap();
            } else {
                eidola_engine_cuda::launch!(
                    gpu,
                    reference.router,
                    [u32_of(tokens), 1, 1],
                    dptr(&ids, s),
                    dptr(&wts, s),
                    dptr(&x, s),
                    dptr(&w, s),
                    dptr(&bias, s),
                    u32_of(HIDDEN),
                    u32_of(experts),
                    u32_of(TOP_K),
                    1.0f32
                )
                .unwrap();
            }
        }
        let ids = s.clone_dtoh(&ids).unwrap();
        let wts: Vec<u32> = s
            .clone_dtoh(&wts)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        out.push((ids, wts));
    }
    let reference_out = out.pop().unwrap();
    (out.pop().unwrap(), reference_out)
}

/// The expert kernels' launch contracts are the geometry the host launches
/// with.
#[test]
fn router_launch_contract() {
    let Some(su) = setup() else { return };
    for &arch in &su.archs {
        let m = su.module("engine_ops", arch);
        let meta = *m.kernel("eidola_router_topk").unwrap().meta();
        assert_eq!(meta.block, [ROUTER_THREADS, 1, 1], "{arch:?}");
        assert_eq!(meta.cluster, [ROUTER_CLUSTER, 1, 1], "{arch:?}");
        let meta = *m.kernel("eidola_router_topk_tiled").unwrap().meta();
        assert_eq!(meta.block, [ROUTER_TILED_THREADS, 1, 1], "{arch:?}");
        assert_eq!(meta.cluster, [ROUTER_CLUSTER, 1, 1], "{arch:?}");
        assert_eq!(meta.dynamic_smem_bytes, 0, "{arch:?}");
        let meta = *m.kernel("eidola_router_scores").unwrap().meta();
        assert_eq!(meta.block, [ROUTER_SCORES_THREADS, 1, 1], "{arch:?}");
        assert_eq!(meta.cluster, [1, 1, 1], "{arch:?}");
        assert_eq!(meta.dynamic_smem_bytes, ROUTER_SCORES_SMEM, "{arch:?}");
        let meta = *m.kernel("eidola_router_select").unwrap().meta();
        assert_eq!(meta.block, [ROUTER_SELECT_WARPS * 32, 1, 1], "{arch:?}");
        assert_eq!(meta.cluster, [1, 1, 1], "{arch:?}");
        assert_eq!(meta.dynamic_smem_bytes, 0, "{arch:?}");
        let meta = *m.kernel("eidola_moe_combine").unwrap().meta();
        assert_eq!(meta.block, [COMBINE_WIDTH / 8, 1, 1], "{arch:?}");
        assert_eq!(meta.cluster, [1, 1, 1], "{arch:?}");
        for k in ["eidola_gather_quant_ue8m0", "eidola_swiglu_quant_fp8_ue8m0"] {
            let meta = *m.kernel(k).unwrap().meta();
            assert_eq!(meta.block, [128, 1, 1], "{k} {arch:?}");
            assert_eq!(meta.cluster, [1, 1, 1], "{k} {arch:?}");
        }
    }
}

/// Same expert ids in the same order and the same weights, bit for bit,
/// ties included, at every token count, in every form.
#[test]
fn router_matches_reference_bit_for_bit() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let reference = Reference::load(&su, arch);
        for &tokens in &TOKENS {
            for form in FORMS {
                let mut rng = Lcg(0x5eed ^ tokens as u64);
                let case = router_case(&mut rng, tokens);
                let (new, old) = run_router(gpu, &ops, form, &reference, &case, tokens);
                for t in 0..tokens {
                    let r = t * TOP_K..(t + 1) * TOP_K;
                    assert_eq!(
                        new.0[r.clone()],
                        old.0[r.clone()],
                        "{arch:?} {form:?} {tokens} tokens, token {t}: ids"
                    );
                    assert_eq!(
                        new.1[r.clone()],
                        old.1[r],
                        "{arch:?} {form:?} {tokens} tokens, token {t}: weights"
                    );
                }
                // The tie cases did tie: a zero token picks the experts with the
                // largest bias, lowest ids first among equals.
                let mut zero_pick: Vec<usize> = (0..EXPERTS).collect();
                zero_pick.sort_by(|&a, &b| case.bias[b].total_cmp(&case.bias[a]).then(a.cmp(&b)));
                let mut want: Vec<i32> = zero_pick[..TOP_K]
                    .iter()
                    .map(|&e| i32::try_from(e).unwrap())
                    .collect();
                want.sort_unstable();
                assert_eq!(
                    new.0[..TOP_K],
                    want[..],
                    "{arch:?} {form:?} {tokens} tokens: zero token"
                );
            }
        }
    }
}

/// Fewer experts than the kernels' widest (a partial last expert block of
/// the split form, and lanes holding no expert in its selection): the same
/// ids and weights as the reference, bit for bit, in every form.
#[test]
fn router_partial_experts_match_reference() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let reference = Reference::load(&su, arch);
        for experts in [200, 250] {
            for tokens in [1, 9, 64, 513] {
                let mut rng = Lcg(0xe0 ^ (experts * 31 + tokens) as u64);
                let case = router_case(&mut rng, tokens);
                for form in FORMS {
                    let (new, old) =
                        run_router_experts(gpu, &ops, form, &reference, &case, tokens, experts);
                    assert_eq!(
                        new, old,
                        "{arch:?} {form:?} {experts} experts, {tokens} tokens"
                    );
                    assert!(
                        new.0
                            .iter()
                            .all(|&e| usize::try_from(e).is_ok_and(|e| e < experts)),
                        "{arch:?} {form:?} {experts} experts, {tokens} tokens: an id out of range"
                    );
                }
            }
        }
    }
}

/// NaN choices select exactly as the reference's scan does: a NaN at the
/// lowest untaken expert is taken, one elsewhere never wins. Every form, at
/// every token count, with NaN logits too (a NaN in a token's row makes every
/// one of its scores NaN).
#[test]
fn router_nan_choices_match_reference() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let reference = Reference::load(&su, arch);
        for &tokens in &TOKENS {
            let mut rng = Lcg(99 ^ tokens as u64);
            let mut case = router_case(&mut rng, tokens);
            case.bias[0] = f32::NAN;
            case.bias[77] = f32::NAN;
            case.bias[200] = f32::INFINITY;
            // Token 3's row holds a NaN; token 4's an infinity.
            if tokens > 4 {
                case.x[3 * HIDDEN + 1000] = f32::NAN;
                case.x[4 * HIDDEN + 17] = f32::INFINITY;
            }
            for form in FORMS {
                let (new, old) = run_router(gpu, &ops, form, &reference, &case, tokens);
                assert_eq!(new, old, "{arch:?} {form:?} {tokens} tokens");
                for t in (0..tokens).filter(|&t| t != 3 || tokens <= 4) {
                    let ids = &new.0[t * TOP_K..(t + 1) * TOP_K];
                    assert!(
                        ids.contains(&0) && ids.contains(&200) && !ids.contains(&77),
                        "{arch:?} {form:?} {tokens} tokens, token {t}: {ids:?}"
                    );
                }
            }
        }
    }
}

/// Against the model crate's routing on inputs whose logits are exact in any
/// summation order (small dyadic values): the same experts in the same order;
/// the weights differ only by the device's `expf`, a few ulp. Tokens whose
/// selection the host decides by a margin within that rounding are not
/// compared, and they must be rare.
#[test]
fn router_matches_model_routing() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let spec = MoeSpec {
        num_experts: EXPERTS,
        top_k: TOP_K,
        intermediate_size: INTER,
        norm_topk_prob: true,
        routed_scaling_factor: 1.0,
        router_dtype: Some("bfloat16".into()),
    };
    let tokens = 513;
    let mut rng = Lcg(4242);
    let tri = |rng: &mut Lcg| match rng.below(3) {
        0 => -1.0f32,
        1 => 0.0,
        _ => 1.0,
    };
    let w: Vec<f32> = (0..EXPERTS * HIDDEN)
        .map(|_| tri(&mut rng) * 0.125)
        .collect();
    let w_bits: Vec<u16> = w.iter().map(|&v| bf16::from_f32(v)).collect();
    let bias: Vec<f32> = (0..EXPERTS)
        .map(|e| ((e * 97) % EXPERTS) as f32 / 1024.0)
        .collect();
    let mut x = vec![0f32; tokens * HIDDEN];
    for t in 1..tokens {
        for v in &mut x[t * HIDDEN..(t + 1) * HIDDEN] {
            if rng.below(8) == 0 {
                *v = tri(&mut rng);
            }
        }
    }
    let router = Matrix::from_vec(EXPERTS, HIDDEN, w.clone());
    let case = RouterCase {
        x: x.clone(),
        w: w_bits,
        bias: bias.clone(),
    };
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let reference = Reference::load(&su, arch);
        for form in FORMS {
            let (new, _) = run_router(gpu, &ops, form, &reference, &case, tokens);
            let mut compared = 0;
            for t in 0..tokens {
                let xr = &x[t * HIDDEN..(t + 1) * HIDDEN];
                // The host's choices, to measure the selection margin.
                let mut choices: Vec<f32> = (0..EXPERTS)
                    .map(|e| {
                        let dot: f32 = xr.iter().zip(router.row(e)).map(|(a, b)| a * b).sum();
                        1.0 / (1.0 + (-dot).exp()) + bias[e]
                    })
                    .collect();
                choices.sort_by(|a, b| b.total_cmp(a));
                let (kth, next) = (choices[TOP_K - 1], choices[TOP_K]);
                if kth != next && kth - next < 1e-5 {
                    continue;
                }
                compared += 1;
                let want = route(&spec, &router, &bias, xr);
                let ids: Vec<usize> = new.0[t * TOP_K..(t + 1) * TOP_K]
                    .iter()
                    .map(|&e| usize::try_from(e).unwrap())
                    .collect();
                assert_eq!(ids, want.experts, "{arch:?} {form:?} token {t}");
                for (j, &bits) in new.1[t * TOP_K..(t + 1) * TOP_K].iter().enumerate() {
                    let (got, w) = (f32::from_bits(bits), want.weights[j]);
                    assert!(
                        (got - w).abs() <= 8.0 * f32::EPSILON * w.abs(),
                        "{arch:?} {form:?} token {t} slot {j}: {got} vs {w}"
                    );
                }
            }
            assert!(
                compared * 10 >= tokens * 9,
                "{arch:?} {form:?}: only {compared} of {tokens} tokens decided by a clear margin"
            );
        }
    }
}

/// Distinct experts per token, skewed toward low ids so some experts take
/// more than one 128-row block in the psum layout.
fn topk_ids(rng: &mut Lcg, tokens: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(tokens * TOP_K);
    for _ in 0..tokens {
        let mut row: Vec<i32> = Vec::with_capacity(TOP_K);
        while row.len() < TOP_K {
            let u = rng.below(EXPERTS as u64);
            let e = i32::try_from(u * u / EXPERTS as u64).unwrap();
            if !row.contains(&e) {
                row.push(e);
            }
        }
        row.sort_unstable();
        ids.extend(row);
    }
    ids
}

/// The executor's expert layout for a step of `tokens` rows.
struct Layout {
    rows: usize,
    rows4: usize,
}

impl Layout {
    fn for_tokens(tokens: usize) -> Layout {
        let rows = psum_rows(tokens, TOP_K);
        Layout {
            rows,
            rows4: rows.div_ceil(4) * 4,
        }
    }

    fn sf_words(&self, k: usize) -> usize {
        k / 512 * self.rows4
    }

    /// `sfa_index` in `engine_ops_common.cuh`.
    fn sf_index(&self, r: usize, w: usize) -> usize {
        w * self.rows4 + r
    }
}

/// The executor's placement of `ids` (`row_of` per pair, read back), checked
/// word for word against its host form along with the grouped layout.
fn place(gpu: &Gpu, ops: &EngineOps, ids: &[i32], tokens: usize, layout: &Layout) -> Vec<i32> {
    let (row_of, grouped) = permute(gpu, ops, ids, tokens, layout);
    let (want_row_of, want_grouped) = expert_placement(ids);
    assert_eq!(row_of, want_row_of, "{tokens} tokens: row_of");
    assert_eq!(grouped, want_grouped, "{tokens} tokens: grouped layout");
    let mut seen = vec![false; layout.rows];
    for &r in &row_of {
        let r = usize::try_from(r).unwrap();
        assert!(
            r < layout.rows && !seen[r],
            "row {r} out of range or taken twice"
        );
        seen[r] = true;
    }
    row_of
}

/// `eidola_moe_permute` over `ids`: `row_of` and the grouped layout, read
/// back; both start as sentinels, so a word the kernel leaves reads -7.
fn permute(
    gpu: &Gpu,
    ops: &EngineOps,
    ids: &[i32],
    tokens: usize,
    layout: &Layout,
) -> (Vec<i32>, Vec<i32>) {
    let s = gpu.stream();
    let dids = s.clone_htod(ids).unwrap();
    let grouped = s.clone_htod(&vec![-7i32; EXPERTS]).unwrap();
    let row_of = s.clone_htod(&vec![-7i32; ids.len()]).unwrap();
    unsafe {
        ops.moe_permute(
            gpu,
            dptr(&grouped, s),
            dptr(&row_of, s),
            dptr(&dids, s),
            u32_of(tokens),
            u32_of(TOP_K),
            u32_of(layout.rows),
        )
        .unwrap();
    }
    (
        s.clone_dtoh(&row_of).unwrap(),
        s.clone_dtoh(&grouped).unwrap(),
    )
}

/// The placement equals its host form word for word at every count, over
/// spread and concentrated routing (runs of many blocks), and an id outside
/// the experts is placed nowhere: its `row_of` word is left as it was and it
/// counts for no expert.
#[test]
fn permute_matches_host_placement() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        for &tokens in &TOKENS {
            let mut rng = Lcg(0x5e7 ^ tokens as u64);
            let spread = topk_ids(&mut rng, tokens);
            // Every token on 8 of the first 12 experts.
            let concentrated: Vec<i32> = (0..tokens)
                .flat_map(|t| (0..TOP_K).map(move |j| i32::try_from((t + 3 * j) % 12).unwrap()))
                .collect();
            let mut stray = spread.clone();
            for i in (0..stray.len()).step_by(37) {
                stray[i] = if i % 2 == 0 { -1 } else { 256 };
            }
            let layout = Layout::for_tokens(tokens);
            for (what, ids) in [
                ("spread", &spread),
                ("concentrated", &concentrated),
                ("stray", &stray),
            ] {
                let (row_of, grouped) = permute(gpu, &ops, ids, tokens, &layout);
                let (want_row_of, want_grouped) = expert_placement(ids);
                let want_row_of: Vec<i32> = want_row_of
                    .iter()
                    .map(|&r| if r < 0 { -7 } else { r })
                    .collect();
                let tag = format!("{arch:?} {tokens} tokens {what}");
                assert_eq!(row_of, want_row_of, "{tag}: row_of");
                assert_eq!(grouped, want_grouped, "{tag}: grouped layout");
            }
        }
    }
}

const SENTINEL_BYTE: u8 = 0xa5;
const SENTINEL_WORD: i32 = 0x5a5a_5a5a;

/// Compares the routed rows of two quantized outputs and their scale words,
/// and checks that the executor's kernel left every other row as it was.
#[allow(clippy::too_many_arguments)]
fn compare_rows(
    what: &str,
    row_of: &[i32],
    layout: &Layout,
    k: usize,
    new_q: &[u8],
    new_sf: &[i32],
    old_q: &[u8],
    old_sf: &[i32],
) {
    let mut routed = vec![false; layout.rows];
    for &r in row_of {
        let r = usize::try_from(r).unwrap();
        routed[r] = true;
        assert_eq!(
            new_q[r * k..(r + 1) * k],
            old_q[r * k..(r + 1) * k],
            "{what}: row {r} codes"
        );
        for w in 0..k / 512 {
            let i = layout.sf_index(r, w);
            assert_eq!(new_sf[i], old_sf[i], "{what}: row {r} scale word {w}");
        }
    }
    for r in (0..layout.rows).filter(|&r| !routed[r]) {
        assert!(
            new_q[r * k..(r + 1) * k]
                .iter()
                .all(|&b| b == SENTINEL_BYTE),
            "{what}: padding row {r} written"
        );
        for w in 0..k / 512 {
            assert_eq!(
                new_sf[layout.sf_index(r, w)],
                SENTINEL_WORD,
                "{what}: padding row {r} scale written"
            );
        }
    }
}

/// A value of magnitude around 2^`e` with a random sign and mantissa.
fn scaled(rng: &mut Lcg, e: i32) -> f32 {
    rng.f32() * 2f32.powi(e)
}

/// The gather quantizes every routed pair's token row into its row exactly as
/// the reference does (codes and scale words), and touches no other row.
#[test]
fn gather_matches_reference_bit_for_bit() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let k = HIDDEN;
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let reference = Reference::load(&su, arch);
        for &tokens in &TOKENS {
            let mut rng = Lcg(0x9a7 ^ tokens as u64);
            let layout = Layout::for_tokens(tokens);
            let ids = topk_ids(&mut rng, tokens);
            let row_of = place(gpu, &ops, &ids, tokens, &layout);
            let mut row_src = vec![-1i32; layout.rows];
            for (i, &r) in row_of.iter().enumerate() {
                row_src[usize::try_from(r).unwrap()] = i32::try_from(i / TOP_K).unwrap();
            }
            // Per 128-group magnitudes from 2^-140 (subnormal, below the
            // smallest scale) to 2^127 (past FP8's range at the largest
            // scale), all-zero groups, and single spikes.
            let mut x = vec![0f32; tokens * k];
            for (gi, group) in x.chunks_mut(128).enumerate() {
                match gi % 9 {
                    0 => {}
                    1 => group[usize::try_from(rng.below(128)).unwrap()] = scaled(&mut rng, 10),
                    _ => {
                        let e = i32::try_from(rng.below(268)).unwrap() - 140;
                        for v in group.iter_mut() {
                            *v = scaled(&mut rng, e);
                        }
                    }
                }
            }
            let dx = s.clone_htod(&x).unwrap();
            let drow_of = s.clone_htod(&row_of).unwrap();
            let drow_src = s.clone_htod(&row_src).unwrap();
            let mut outs = Vec::new();
            for new in [true, false] {
                let q = s.clone_htod(&vec![SENTINEL_BYTE; layout.rows * k]).unwrap();
                let sf = s
                    .clone_htod(&vec![SENTINEL_WORD; layout.sf_words(k)])
                    .unwrap();
                unsafe {
                    if new {
                        ops.gather_quant_ue8m0(
                            gpu,
                            dptr(&q, s),
                            dptr(&sf, s),
                            dptr(&dx, s),
                            dptr(&drow_of, s),
                            u32_of(tokens),
                            u32_of(TOP_K),
                            u32_of(layout.rows),
                            u32_of(k),
                            u32_of(layout.rows4),
                        )
                        .unwrap();
                    } else {
                        eidola_engine_cuda::launch!(
                            gpu,
                            reference.gather,
                            [u32_of(layout.rows), u32_of(k / 512), 1],
                            dptr(&q, s),
                            dptr(&sf, s),
                            dptr(&dx, s),
                            dptr(&drow_src, s),
                            u32_of(k),
                            u32_of(layout.rows4)
                        )
                        .unwrap();
                    }
                }
                outs.push((s.clone_dtoh(&q).unwrap(), s.clone_dtoh(&sf).unwrap()));
            }
            compare_rows(
                &format!("gather {arch:?} {tokens} tokens"),
                &row_of,
                &layout,
                k,
                &outs[0].0,
                &outs[0].1,
                &outs[1].0,
                &outs[1].1,
            );
        }
    }
}

/// SwiGLU + quantization of every routed row exactly as the reference does,
/// touching no other row.
#[test]
fn swiglu_matches_reference_bit_for_bit() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let reference = Reference::load(&su, arch);
        for &tokens in &TOKENS {
            let mut rng = Lcg(0x5719 ^ tokens as u64);
            let layout = Layout::for_tokens(tokens);
            let ids = topk_ids(&mut rng, tokens);
            let row_of = place(gpu, &ops, &ids, tokens, &layout);
            // Gate/up rows of the grouped GEMM's output: magnitudes from 2^-8
            // to 2^8 per row (exp overflows and underflows in SiLU), plus a
            // zero row and BF16 infinities and NaN.
            let mut gu = vec![0u16; layout.rows * 2 * INTER];
            for (i, &r) in row_of.iter().enumerate() {
                let row = &mut gu[usize::try_from(r).unwrap() * 2 * INTER..][..2 * INTER];
                if i % 11 == 3 {
                    continue;
                }
                let e = i32::try_from(rng.below(17)).unwrap() - 8;
                for v in row.iter_mut() {
                    *v = bf16::from_f32(scaled(&mut rng, e));
                }
                if i % 13 == 5 {
                    row[7] = bf16::from_f32(f32::INFINITY);
                    row[INTER + 300] = bf16::from_f32(f32::NEG_INFINITY);
                    row[1000] = bf16::from_f32(f32::NAN);
                }
            }
            let dgu = s.clone_htod(&gu).unwrap();
            let drow_of = s.clone_htod(&row_of).unwrap();
            let mut outs = Vec::new();
            for new in [true, false] {
                let q = s
                    .clone_htod(&vec![SENTINEL_BYTE; layout.rows * INTER])
                    .unwrap();
                let sf = s
                    .clone_htod(&vec![SENTINEL_WORD; layout.sf_words(INTER)])
                    .unwrap();
                unsafe {
                    if new {
                        ops.swiglu_quant_ue8m0(
                            gpu,
                            dptr(&q, s),
                            dptr(&sf, s),
                            dptr(&dgu, s),
                            dptr(&drow_of, s),
                            u32_of(tokens),
                            u32_of(TOP_K),
                            u32_of(layout.rows),
                            u32_of(INTER),
                            u32_of(layout.rows4),
                        )
                        .unwrap();
                    } else {
                        eidola_engine_cuda::launch!(
                            gpu,
                            reference.swiglu,
                            [u32_of(layout.rows), u32_of(INTER / 512), 1],
                            dptr(&q, s),
                            dptr(&sf, s),
                            dptr(&dgu, s),
                            u32_of(INTER),
                            u32_of(layout.rows4)
                        )
                        .unwrap();
                    }
                }
                outs.push((s.clone_dtoh(&q).unwrap(), s.clone_dtoh(&sf).unwrap()));
            }
            compare_rows(
                &format!("swiglu {arch:?} {tokens} tokens"),
                &row_of,
                &layout,
                INTER,
                &outs[0].0,
                &outs[0].1,
                &outs[1].0,
                &outs[1].1,
            );
        }
    }
}

/// The combine sums every token's k routed rows in ascending slot order
/// exactly as the reference does, over rows with infinities, NaN, zero
/// weights and magnitudes whose sums round differently in another order.
#[test]
fn combine_matches_reference_bit_for_bit() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let reference = Reference::load(&su, arch);
        for &tokens in &TOKENS {
            let mut rng = Lcg(0xc0b1 ^ tokens as u64);
            let layout = Layout::for_tokens(tokens);
            let ids = topk_ids(&mut rng, tokens);
            let row_of = place(gpu, &ops, &ids, tokens, &layout);
            let mut d = vec![0u16; layout.rows * HIDDEN];
            for (i, &r) in row_of.iter().enumerate() {
                let row = &mut d[usize::try_from(r).unwrap() * HIDDEN..][..HIDDEN];
                let e = i32::try_from(rng.below(41)).unwrap() - 20;
                for v in row.iter_mut() {
                    *v = bf16::from_f32(scaled(&mut rng, e));
                }
                if i % 17 == 4 {
                    row[5] = bf16::from_f32(f32::INFINITY);
                    row[6] = bf16::from_f32(f32::NEG_INFINITY);
                    row[2047] = bf16::from_f32(f32::NAN);
                }
            }
            let w: Vec<f32> = (0..tokens * TOP_K)
                .map(|i| {
                    if i % 23 == 9 {
                        0.0
                    } else {
                        rng.f32().abs() * 0.5
                    }
                })
                .collect();
            let dd = s.clone_htod(&d).unwrap();
            let dw = s.clone_htod(&w).unwrap();
            let drow_of = s.clone_htod(&row_of).unwrap();
            let mut outs = Vec::new();
            for new in [true, false] {
                let out = s.clone_htod(&vec![f32::NAN; tokens * HIDDEN]).unwrap();
                unsafe {
                    if new {
                        ops.moe_combine(
                            gpu,
                            dptr(&out, s),
                            dptr(&dd, s),
                            dptr(&drow_of, s),
                            dptr(&dw, s),
                            u32_of(tokens),
                            u32_of(HIDDEN),
                            u32_of(TOP_K),
                        )
                        .unwrap();
                    } else {
                        eidola_engine_cuda::launch!(
                            gpu,
                            reference.combine,
                            [u32_of(tokens), 1, 1],
                            dptr(&out, s),
                            dptr(&dd, s),
                            dptr(&drow_of, s),
                            dptr(&dw, s),
                            u32_of(HIDDEN),
                            u32_of(TOP_K)
                        )
                        .unwrap();
                    }
                }
                let bits: Vec<u32> = s
                    .clone_dtoh(&out)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                outs.push(bits);
            }
            for t in 0..tokens {
                let r = t * HIDDEN..(t + 1) * HIDDEN;
                assert_eq!(
                    outs[0][r.clone()],
                    outs[1][r],
                    "{arch:?} {tokens} tokens, token {t}"
                );
            }
        }
    }
}
