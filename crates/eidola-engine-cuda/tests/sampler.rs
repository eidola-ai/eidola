//! The device sampler against the serving core's reference: every processed
//! probability and every token bit for bit, on every image this device runs.

mod common;

use common::{Lcg, setup};
use eidola_engine::sampling::{self, Logits, SamplingParams, Stream};
use eidola_engine_cuda::sampler::{AcceptInputs, AcceptRow, STATUS_NON_FINITE, SampleRow, Sampler};

/// MiMo's padded head and tokenizer.
const STRIDE: usize = 152_576;
const SAMPLEABLE: u32 = 151_675;

/// Logit rows of several shapes: smooth, peaked, heavily tied (values on a
/// coarse grid, so top-k, top-p and min-p all cut through ties), with masked
/// (-inf) entries, and with padded entries that would win if they were
/// sampleable.
fn logit_rows(rng: &mut Lcg, rows: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * STRIDE];
    for r in 0..rows {
        let row = &mut out[r * STRIDE..(r + 1) * STRIDE];
        let kind = r % 5;
        for (i, v) in row.iter_mut().enumerate() {
            let x = rng.f32();
            *v = match kind {
                0 => x * 4.0,
                1 => x * 2.0 + if i % 9973 == 17 { 12.0 } else { 0.0 },
                2 => (x * 6.0).round() * 0.5,
                3 => {
                    if rng.below(7) == 0 {
                        f32::NEG_INFINITY
                    } else {
                        x * 8.0
                    }
                }
                _ => x * 20.0,
            };
        }
        for v in &mut row[SAMPLEABLE as usize..] {
            *v = 100.0;
        }
    }
    out
}

fn params_set() -> Vec<SamplingParams> {
    let p = |t, k, tp, mp, seed| SamplingParams::new(t, k, tp, mp, seed).unwrap();
    vec![
        SamplingParams::greedy(),
        p(1.0, 0, 1.0, 0.0, 1),
        p(1.0, 0, 0.95, 0.0, 2),
        p(0.7, 0, 0.9, 0.0, 3),
        p(1.0, 50, 1.0, 0.0, 4),
        p(0.8, 40, 0.9, 0.05, 5),
        p(1.0, 0, 1.0, 0.1, 6),
        p(1.3, 0, 0.5, 0.0, 7),
        p(1.0, 1, 1.0, 0.0, 8),
        p(2.0, 200_000, 0.99, 0.0, 9),
        p(0.05, 0, 1.0, 0.0, 10),
    ]
}

fn reference_token(logits: Logits<'_>, params: &SamplingParams, pos: u32, stream: Stream) -> u32 {
    if params.is_greedy() {
        return sampling::argmax(logits);
    }
    let probs = sampling::processed_probs(logits, params);
    sampling::sample_from(&probs, sampling::uniform(params.seed(), pos as u64, stream))
}

#[test]
fn sample_matches_reference_bit_for_bit() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let mut rng = Lcg(7);
    let num_logit_rows = 10;
    let logits = logit_rows(&mut rng, num_logit_rows);
    let dlogits = s.clone_htod(&logits).unwrap();
    let params = params_set();
    // Every parameter set against every logit row, at a few positions.
    let mut rows = Vec::new();
    for (pi, p) in params.iter().enumerate() {
        for lr in 0..num_logit_rows {
            let pos = (pi * 31 + lr * 7) as u32 + 1;
            rows.push((SampleRow::new(p, pos, lr as u32), *p));
        }
    }
    let n = SAMPLEABLE as usize;
    let hrows: Vec<SampleRow> = rows.iter().map(|r| r.0).collect();
    let mut drows = s.clone_htod(&hrows).unwrap();
    for &arch in &su.archs {
        let sampler = Sampler::from_module(su.module("sampling", arch)).unwrap();
        for stream in [Stream::Sample, Stream::Draft] {
            let mut probs = s.alloc_zeros::<f64>(rows.len() * n).unwrap();
            let mut tokens = s.alloc_zeros::<u32>(rows.len()).unwrap();
            let mut status = s.alloc_zeros::<u32>(1).unwrap();
            let t0 = std::time::Instant::now();
            sampler
                .sample(
                    gpu,
                    &dlogits,
                    STRIDE,
                    SAMPLEABLE,
                    &hrows,
                    &mut drows,
                    Some(stream),
                    &mut probs,
                    &mut tokens,
                    &mut status,
                )
                .unwrap();
            gpu.synchronize().unwrap();
            eprintln!(
                "{arch:?} {stream:?}: {} rows in {:?}",
                rows.len(),
                t0.elapsed()
            );
            let probs = s.clone_dtoh(&probs).unwrap();
            let tokens = s.clone_dtoh(&tokens).unwrap();
            assert_eq!(s.clone_dtoh(&status).unwrap()[0], 0);
            for (r, (row, params)) in rows.iter().enumerate() {
                let lr = row.logit_row as usize;
                let lrow = Logits::new(&logits[lr * STRIDE..(lr + 1) * STRIDE], SAMPLEABLE);
                let want = sampling::processed_probs(lrow, params);
                let got = &probs[r * n..(r + 1) * n];
                let bad = want
                    .iter()
                    .zip(got)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                assert!(
                    bad.is_none(),
                    "{arch:?} row {r} ({params:?}, logits {lr}): prob {} differs: {:e} vs {:e}",
                    bad.unwrap(),
                    want[bad.unwrap()],
                    got[bad.unwrap()]
                );
                let want_token = reference_token(lrow, params, row.position, stream);
                assert_eq!(
                    tokens[r], want_token,
                    "{arch:?} row {r} ({params:?}, logits {lr}, {stream:?})"
                );
                assert!(tokens[r] < SAMPLEABLE);
            }
        }
    }
}

/// Many draws from one row: the device inverse CDF agrees with the reference
/// across the whole [0, 1) range of uniforms, not just at a few positions.
#[test]
fn draws_match_reference_over_many_positions() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let mut rng = Lcg(11);
    let logits = logit_rows(&mut rng, 1);
    let dlogits = s.clone_htod(&logits).unwrap();
    let params = SamplingParams::new(1.0, 0, 0.9, 0.0, 1234).unwrap();
    let rows: Vec<SampleRow> = (0..512).map(|i| SampleRow::new(&params, i, 0)).collect();
    let mut drows = s.clone_htod(&rows).unwrap();
    let n = SAMPLEABLE as usize;
    let lrow = Logits::new(&logits[..STRIDE], SAMPLEABLE);
    let probs_ref = sampling::processed_probs(lrow, &params);
    for &arch in &su.archs {
        let sampler = Sampler::from_module(su.module("sampling", arch)).unwrap();
        let mut probs = s.alloc_zeros::<f64>(rows.len() * n).unwrap();
        let mut tokens = s.alloc_zeros::<u32>(rows.len()).unwrap();
        let mut status = s.alloc_zeros::<u32>(1).unwrap();
        sampler
            .sample(
                gpu,
                &dlogits,
                STRIDE,
                SAMPLEABLE,
                &rows,
                &mut drows,
                Some(Stream::Sample),
                &mut probs,
                &mut tokens,
                &mut status,
            )
            .unwrap();
        let tokens = s.clone_dtoh(&tokens).unwrap();
        for (i, row) in rows.iter().enumerate() {
            let u = sampling::uniform(params.seed(), row.position as u64, Stream::Sample);
            assert_eq!(
                tokens[i],
                sampling::sample_from(&probs_ref, u),
                "{arch:?} position {}",
                row.position
            );
        }
    }
}

#[test]
fn non_finite_logits_are_reported() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let n = 4096usize;
    for (bad, temperature) in [(f32::NAN, 0.0), (f32::INFINITY, 1.0), (f32::NAN, 1.0)] {
        let mut logits = vec![0.5f32; n];
        logits[100] = bad;
        let dlogits = s.clone_htod(&logits).unwrap();
        let params = SamplingParams::random(temperature, 1).unwrap();
        let hrow = [SampleRow::new(&params, 3, 0)];
        let mut drows = s.clone_htod(&hrow).unwrap();
        for &arch in &su.archs {
            let sampler = Sampler::from_module(su.module("sampling", arch)).unwrap();
            let mut probs = s.alloc_zeros::<f64>(n).unwrap();
            let mut tokens = s.alloc_zeros::<u32>(1).unwrap();
            let mut status = s.alloc_zeros::<u32>(1).unwrap();
            sampler
                .sample(
                    gpu,
                    &dlogits,
                    n,
                    n as u32,
                    &hrow,
                    &mut drows,
                    Some(Stream::Sample),
                    &mut probs,
                    &mut tokens,
                    &mut status,
                )
                .unwrap();
            assert_eq!(
                s.clone_dtoh(&status).unwrap()[0] & STATUS_NON_FINITE,
                STATUS_NON_FINITE,
                "{bad} at temperature {temperature}"
            );
        }
    }
}

/// Chain acceptance (accept draws, residual draws, bonus draws, greedy) against
/// `sampling::chain_accept` on distributions the device computed (and that
/// equal the reference's, per the test above).
#[test]
fn chain_accept_matches_reference() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let mut rng = Lcg(23);
    let n = 3000usize;
    let stride = 4u32;
    let params = [
        SamplingParams::greedy(),
        SamplingParams::random(1.0, 5).unwrap(),
        SamplingParams::new(0.8, 0, 0.9, 0.0, 6).unwrap(),
        SamplingParams::new(1.0, 20, 1.0, 0.02, 7).unwrap(),
    ];
    // Per sequence: k drafts; target rows k+1 and draft rows k, each a logit
    // row close to the target's so acceptance is neither certain nor rare.
    let mut seqs = Vec::new();
    let (mut target_logits, mut draft_logits) = (Vec::new(), Vec::new());
    for i in 0..48usize {
        let k = 1 + i % 3;
        let p = params[i % params.len()];
        let tr = target_logits.len() / n;
        let dr = draft_logits.len() / n;
        for _ in 0..=k {
            let base: Vec<f32> = (0..n).map(|_| rng.f32() * 3.0).collect();
            target_logits.extend(&base);
        }
        for j in 0..k {
            let t = &target_logits[(tr + j) * n..(tr + j + 1) * n];
            let noisy: Vec<f32> = t
                .iter()
                .map(|&v| if i % 4 == 3 { v } else { v + rng.f32() * 0.7 })
                .collect();
            draft_logits.extend(noisy);
        }
        seqs.push((k, p, tr, dr, (i * 13) as u32 + 1));
    }
    let nt = target_logits.len() / n;
    let nd = draft_logits.len() / n;
    let t_rows: Vec<SampleRow> = (0..nt)
        .map(|r| {
            let (_, p, ..) = seqs.iter().find(|q| r >= q.2 && r <= q.2 + q.0).unwrap();
            SampleRow::new(p, 0, r as u32)
        })
        .collect();
    let d_rows: Vec<SampleRow> = (0..nd)
        .map(|r| {
            let (_, p, ..) = seqs.iter().find(|q| r >= q.3 && r < q.3 + q.0).unwrap();
            SampleRow::new(p, 0, r as u32)
        })
        .collect();
    // Drafts: drawn on the host from the reference draft distributions.
    let mut drafts = vec![0u32; seqs.len() * stride as usize];
    let mut q_ref = Vec::new();
    let mut p_ref = Vec::new();
    for (si, &(k, p, tr, dr, pos)) in seqs.iter().enumerate() {
        let qs: Vec<Vec<f64>> = (0..k)
            .map(|j| {
                let l = Logits::new(&draft_logits[(dr + j) * n..(dr + j + 1) * n], n as u32);
                sampling::processed_probs(l, &p)
            })
            .collect();
        for (j, q) in qs.iter().enumerate() {
            drafts[si * stride as usize + j] = sampling::sample_from(
                q,
                sampling::uniform(p.seed(), (pos + j as u32) as u64, Stream::Draft),
            );
        }
        let ps: Vec<Vec<f64>> = (0..=k)
            .map(|j| {
                let l = Logits::new(&target_logits[(tr + j) * n..(tr + j + 1) * n], n as u32);
                sampling::processed_probs(l, &p)
            })
            .collect();
        q_ref.push(qs);
        p_ref.push(ps);
    }
    let dt = s.clone_htod(&target_logits).unwrap();
    let dd = s.clone_htod(&draft_logits).unwrap();
    let mut dtr = s.clone_htod(&t_rows).unwrap();
    let mut ddr = s.clone_htod(&d_rows).unwrap();
    let rows: Vec<SampleRow> = seqs
        .iter()
        .map(|&(_, p, _, _, pos)| SampleRow::new(&p, pos, 0))
        .collect();
    let drows = s.clone_htod(&rows).unwrap();
    let plan: Vec<AcceptRow> = seqs
        .iter()
        .enumerate()
        .map(|(si, &(k, _, tr, dr, _))| AcceptRow {
            target_row: tr as u32,
            draft_row: dr as u32,
            drafts: &drafts[si * stride as usize..][..k],
        })
        .collect();
    let mut inputs = AcceptInputs::new(gpu, seqs.len(), stride).unwrap();
    for &arch in &su.archs {
        let sampler = Sampler::from_module(su.module("sampling", arch)).unwrap();
        let mut tp = s.alloc_zeros::<f64>(nt * n).unwrap();
        let mut dp = s.alloc_zeros::<f64>(nd * n).unwrap();
        let mut tok = s.alloc_zeros::<u32>(nt.max(nd)).unwrap();
        let mut status = s.alloc_zeros::<u32>(1).unwrap();
        sampler
            .sample(
                gpu,
                &dt,
                n,
                n as u32,
                &t_rows,
                &mut dtr,
                None,
                &mut tp,
                &mut tok,
                &mut status,
            )
            .unwrap();
        sampler
            .sample(
                gpu,
                &dd,
                n,
                n as u32,
                &d_rows,
                &mut ddr,
                None,
                &mut dp,
                &mut tok,
                &mut status,
            )
            .unwrap();
        let mut scratch = s.alloc_zeros::<f64>(seqs.len() * n).unwrap();
        let mut out = s.alloc_zeros::<u32>(seqs.len() * stride as usize).unwrap();
        let mut counts = s.alloc_zeros::<u32>(seqs.len()).unwrap();
        sampler
            .chain_accept(
                gpu,
                &tp,
                &dp,
                n as u32,
                &drows,
                &plan,
                &mut inputs,
                &mut scratch,
                &mut out,
                &mut counts,
            )
            .unwrap();
        let out = s.clone_dtoh(&out).unwrap();
        let counts = s.clone_dtoh(&counts).unwrap();
        let mut accepted = 0;
        for (si, &(k, p, _, _, pos)) in seqs.iter().enumerate() {
            let d = &drafts[si * stride as usize..si * stride as usize + k];
            let want = sampling::chain_accept(&p_ref[si], &q_ref[si], d, &p, pos as u64);
            let got = &out[si * stride as usize..si * stride as usize + counts[si] as usize];
            assert_eq!(got, &want[..], "{arch:?} sequence {si} (k {k}, {p:?})");
            accepted += want.len() - 1;
        }
        eprintln!("{arch:?}: {accepted} drafts accepted");
    }
}

/// `-0.0` and `+0.0` are one value: a row whose maximum is zero, held as
/// `-0.0` at a low id and `+0.0` at a higher one in another warp (and the
/// other way round), samples the lower id greedily, as the reference does.
#[test]
fn signed_zero_maxima_tie_to_the_lower_id() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let n = 4096usize;
    let mut logits = vec![-1.0f32; 2 * n];
    logits[5] = -0.0;
    logits[3000] = 0.0;
    logits[n + 5] = 0.0;
    logits[n + 3000] = -0.0;
    for row in 0..2 {
        let r = &logits[row * n..(row + 1) * n];
        assert_eq!(sampling::argmax(Logits::new(r, n as u32)), 5);
    }
    let dlogits = s.clone_htod(&logits).unwrap();
    let rows = [
        SampleRow::new(&SamplingParams::greedy(), 1, 0),
        SampleRow::new(&SamplingParams::greedy(), 1, 1),
    ];
    let mut drows = s.clone_htod(&rows).unwrap();
    for &arch in &su.archs {
        let sampler = Sampler::from_module(su.module("sampling", arch)).unwrap();
        let mut probs = s.alloc_zeros::<f64>(2 * n).unwrap();
        let mut tokens = s.alloc_zeros::<u32>(2).unwrap();
        let mut status = s.alloc_zeros::<u32>(1).unwrap();
        sampler
            .sample(
                gpu,
                &dlogits,
                n,
                n as u32,
                &rows,
                &mut drows,
                Some(Stream::Sample),
                &mut probs,
                &mut tokens,
                &mut status,
            )
            .unwrap();
        assert_eq!(s.clone_dtoh(&tokens).unwrap(), vec![5, 5], "{arch:?}");
    }
}

/// The chained sums hold at most 1,024 chunks of 1,024: a larger vocabulary
/// is refused on the host, for acceptance as for sampling.
#[test]
fn oversized_vocabularies_are_refused() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let sampler = Sampler::from_module(su.module("sampling", su.archs[0])).unwrap();
    let n = (1u32 << 20) + 1;
    let probs = s.alloc_zeros::<f64>(n as usize).unwrap();
    let hrows = [SampleRow::default()];
    let mut rows = s.clone_htod(&hrows).unwrap();
    let plan = [AcceptRow {
        target_row: 0,
        draft_row: 0,
        drafts: &[],
    }];
    let mut inputs = AcceptInputs::new(gpu, 1, 2).unwrap();
    let mut scratch = s.alloc_zeros::<f64>(n as usize).unwrap();
    let mut out = s.alloc_zeros::<u32>(2).unwrap();
    let mut counts = s.alloc_zeros::<u32>(1).unwrap();
    let e = sampler
        .chain_accept(
            gpu,
            &probs,
            &probs,
            n,
            &rows,
            &plan,
            &mut inputs,
            &mut scratch,
            &mut out,
            &mut counts,
        )
        .unwrap_err();
    assert!(e.to_string().contains("vocabulary"), "{e}");
    let logits = s.alloc_zeros::<f32>(n as usize).unwrap();
    let mut tokens = s.alloc_zeros::<u32>(1).unwrap();
    let mut status = s.alloc_zeros::<u32>(1).unwrap();
    let mut p = s.alloc_zeros::<f64>(n as usize).unwrap();
    let e = sampler
        .sample(
            gpu,
            &logits,
            n as usize,
            n,
            &hrows,
            &mut rows,
            None,
            &mut p,
            &mut tokens,
            &mut status,
        )
        .unwrap_err();
    assert!(e.to_string().contains("vocabulary"), "{e}");
}

/// Every index chain acceptance derives is bounded on the host before the
/// launch: a row with as many drafts as its output has slots (no room for
/// the bonus token), target or draft rows past the distributions (or past
/// `u32`, which the kernel adds in), and a draft outside the vocabulary are
/// each refused, and a plan at every limit runs.
#[test]
fn accept_plans_are_bounded() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let sampler = Sampler::from_module(su.module("sampling", su.archs[0])).unwrap();
    let (n, stride) = (64u32, 3u32);
    // Two sequences' worth: 2 x 3 target rows, 2 x 2 draft rows.
    let mut target = vec![0f64; 6 * n as usize];
    for r in 0..6 {
        target[r * n as usize + 7] = 1.0;
    }
    let target = s.clone_htod(&target).unwrap();
    let draft = s.clone_htod(&vec![1.0 / n as f64; 4 * n as usize]).unwrap();
    let rows = s.clone_htod(&[SampleRow::default(); 2]).unwrap();
    let mut inputs = AcceptInputs::new(gpu, 2, stride).unwrap();
    let mut scratch = s.alloc_zeros::<f64>(2 * n as usize).unwrap();
    let mut out = s.alloc_zeros::<u32>(2 * stride as usize).unwrap();
    let mut counts = s.alloc_zeros::<u32>(2).unwrap();
    let mut run = |plan: &[AcceptRow]| {
        sampler.chain_accept(
            gpu,
            &target,
            &draft,
            n,
            &rows,
            plan,
            &mut inputs,
            &mut scratch,
            &mut out,
            &mut counts,
        )?;
        Ok::<_, eidola_engine_cuda::CudaError>((
            s.clone_dtoh(&out).unwrap(),
            s.clone_dtoh(&counts).unwrap(),
        ))
    };
    let row = |target_row, draft_row, drafts| AcceptRow {
        target_row,
        draft_row,
        drafts,
    };
    // At every limit: two drafts in three slots, the last rows of both.
    let (out, counts) = run(&[row(0, 0, &[7, 7]), row(3, 2, &[7, 7])]).unwrap();
    assert_eq!((out, counts), (vec![7; 6], vec![3, 3]));
    let outside = [7, n];
    for (plan, why) in [
        (vec![row(0, 0, &[7, 7]), row(2, 1, &[7, 7, 7])], "drafts"),
        (vec![row(0, 0, &[7, 7]), row(4, 2, &[7, 7])], "target rows"),
        (vec![row(0, 0, &[7]), row(6, 2, &[])], "target rows"),
        (vec![row(0, 0, &[7, 7]), row(3, 3, &[7, 7])], "draft rows"),
        (vec![row(u32::MAX, 0, &[7])], "target rows"),
        (vec![row(0, u32::MAX, &[7])], "draft rows"),
        (vec![row(0, 0, &outside)], "drafts token"),
        (vec![row(0, 0, &[]); 3], "too small"),
    ] {
        let e = run(&plan).unwrap_err();
        assert!(e.to_string().contains(why), "{why}: {e}");
    }
    // A row with no drafts needs no draft rows at all.
    run(&[row(5, u32::MAX, &[])]).unwrap();
}

/// A row naming a logit row outside the logits is refused on the host,
/// before anything is uploaded or launched.
#[test]
fn logit_rows_are_bounded() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    let sampler = Sampler::from_module(su.module("sampling", su.archs[0])).unwrap();
    let n = 4096usize;
    let logits = s.alloc_zeros::<f32>(2 * n).unwrap();
    let mut rows_dev = s.alloc_zeros::<SampleRow>(1).unwrap();
    let mut probs = s.alloc_zeros::<f64>(n).unwrap();
    let mut tokens = s.alloc_zeros::<u32>(1).unwrap();
    let mut status = s.alloc_zeros::<u32>(1).unwrap();
    let mut run = |row: u32| {
        sampler.sample(
            gpu,
            &logits,
            n,
            n as u32,
            &[SampleRow::new(&SamplingParams::greedy(), 1, row)],
            &mut rows_dev,
            None,
            &mut probs,
            &mut tokens,
            &mut status,
        )
    };
    run(1).unwrap();
    let e = run(2).unwrap_err();
    assert!(e.to_string().contains("logit row 2 of 2"), "{e}");
}
