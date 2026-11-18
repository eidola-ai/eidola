//! The Qwen3.8 CUDA executor in its exact profile on host memory, every launch the
//! kernels' own bodies compiled for the host, held to the CPU reference executor step for
//! step through the real engine (`qwen_common::pair`, `qwen_common::scenarios`): tokens,
//! logits bits, state images bytes and encryption, and the memory scan after every step;
//! then reproducibility between calls (`qwen_common::invariance`). Skipped, saying why,
//! without a C++ compiler.

mod qwen_common;

use qwen_common::scenarios;

#[test]
fn one_request_matches_the_reference() {
    if let Some(dev) = qwen_common::compiled() {
        scenarios::one_request(dev);
    }
}

#[test]
fn requests_match_the_reference_at_every_chunking_and_strategy() {
    if let Some(lib) = qwen_common::library() {
        scenarios::chunkings(&|| qwen_common::host_device(lib.clone()));
    }
}

#[test]
fn prefix_hits_preemption_and_the_host_tier_match_the_reference() {
    if let Some(lib) = qwen_common::library() {
        scenarios::hits_and_pressure(&|| qwen_common::host_device(lib.clone()));
    }
}

#[test]
fn one_token_blocks_without_drafts_keep_no_tail() {
    if let Some(lib) = qwen_common::library() {
        scenarios::empty_tails(&|| qwen_common::host_device(lib.clone()));
    }
}

#[test]
fn a_request_alone_and_beside_others_gives_the_same_tokens() {
    if let Some(lib) = qwen_common::library() {
        scenarios::alone_and_crowded(&|| qwen_common::host_device(lib.clone()));
    }
}

#[test]
fn drafts_accepted_in_every_pattern_match_the_reference() {
    if let Some(lib) = qwen_common::library() {
        scenarios::draft_patterns(&|| qwen_common::host_device(lib.clone()));
    }
}

#[test]
fn endings_inside_accepted_drafts_match_the_reference() {
    if let Some(lib) = qwen_common::library() {
        scenarios::endings_inside_drafts(&|| qwen_common::host_device(lib.clone()));
    }
}

/// Reproducibility between calls in the exact profile on the host-compiled kernels: the
/// contract the fast profile is asserted against on a GPU (`tests/qwen_fast.rs`).
#[test]
fn invariance_holds_on_the_compiled_kernels() {
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let make = || qwen_common::host_device(lib.clone());
    let mut failed = Vec::new();
    for (name, r) in qwen_common::invariance::all(&make) {
        match r {
            Ok(what) => eprintln!("ok      {name}: {what}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// The real-weights measurement's path (`QwenCudaExecutor::dense_logits`) against the
/// model crate's forward over one sequence, every position, bit for bit; the slot and
/// blocks it borrowed left zero.
#[test]
fn dense_logits_equal_the_reference_forward() {
    use eidola_engine_cuda::qwen::QwenCudaExecutor;
    use eidola_engine_model::qwen::{LogitRows, QwenForwardOptions};
    use qwen_common::pair::*;
    let Some(dev) = qwen_common::compiled() else {
        return;
    };
    let model = fixture();
    let mut ex =
        QwenCudaExecutor::new(dev, &model, cuda_config(&Geometry::plain(0))).expect("executor");
    let tokens = TestRng(5).tokens(37, VOCAB);
    let got = ex.dense_logits(3, &tokens).expect("logits");
    let opts = QwenForwardOptions {
        logits: LogitRows::All,
        keep_streams: false,
    };
    let want = model
        .forward(&mut model.new_state(), &tokens, &opts)
        .expect("forward");
    for (q, row) in got.iter().enumerate() {
        let w: Vec<u32> = want.logits.row(q).iter().map(|v| v.to_bits()).collect();
        let g: Vec<u32> = row.iter().map(|v| v.to_bits()).collect();
        assert_eq!(g, w, "position {q}");
    }
    assert!(ex.slot_state_is_zero(3));
    assert!((1..=10).all(|b| ex.block_is_zero(0, b)));
}

/// The blocked DeltaNet's distance from the model crate's forward over one sequence of
/// 160 positions at blocks of 16 (`dense_logits`, every other launch the exact profile's):
/// the mean and largest KL divergence of the softmax over the sampleable ids and the top-1
/// agreement, bounded. The fixture's random weights amplify any difference: moving one
/// exact DeltaNet output in eight by a BF16 unit measured KL 1.7e-3 and top-1 141/160
/// here, the blocked form 1.4e-3 and 147/160.
#[test]
fn blocked_deltanet_stays_near_the_reference_forward() {
    use eidola_engine_cuda::qwen::QwenCudaExecutor;
    use eidola_engine_model::qwen::{LogitRows, QwenForwardOptions};
    use qwen_common::pair::*;
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let model = fixture();
    let tokens = TestRng(6).tokens(160, VOCAB);
    let got = qwen_common::with_block_size(16, || {
        let mut ex = QwenCudaExecutor::new(
            qwen_common::block_deltanet(lib),
            &model,
            cuda_config(&Geometry::plain(0)),
        )
        .expect("executor");
        ex.dense_logits(3, &tokens).expect("logits")
    });
    let opts = QwenForwardOptions {
        logits: LogitRows::All,
        keep_streams: false,
    };
    let want = model
        .forward(&mut model.new_state(), &tokens, &opts)
        .expect("forward");
    let n = (SAMPLEABLE as usize).min(model.config().vocab_size);
    let softmax = |x: &[f32]| -> Vec<f64> {
        let m = x[..n].iter().fold(f64::MIN, |a, &b| a.max(f64::from(b)));
        let e: Vec<f64> = x[..n].iter().map(|&v| (f64::from(v) - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    };
    let argmax = |x: &[f32]| {
        (0..n)
            .max_by(|&i, &j| x[i].total_cmp(&x[j]).then(j.cmp(&i)))
            .unwrap_or(0)
    };
    let (mut kl_sum, mut kl_max, mut top) = (0.0f64, 0.0f64, 0usize);
    for (q, row) in got.iter().enumerate() {
        let w = want.logits.row(q);
        let (p, r) = (softmax(w), softmax(row));
        let kl: f64 = p
            .iter()
            .zip(&r)
            .map(|(&a, &b)| {
                if a > 0.0 {
                    a * (a / b.max(1e-300)).ln()
                } else {
                    0.0
                }
            })
            .sum();
        kl_sum += kl;
        kl_max = kl_max.max(kl);
        top += usize::from(argmax(w) == argmax(row));
    }
    let kl_mean = kl_sum / got.len() as f64;
    eprintln!(
        "blocked against the forward over {} positions: KL mean {kl_mean:.3e}, largest \
         {kl_max:.3e}, top-1 {top}/{}",
        got.len(),
        got.len()
    );
    assert!(kl_mean < 3e-3 && top * 100 >= got.len() * 85);
}

/// With the DeltaNet in blocks, `dense_logits` cuts its pieces at whole blocks: a scratch
/// of fewer rows than a block (one row, at blocks of 16) refuses it, before anything runs,
/// rather than cutting a piece past its rows.
#[test]
fn blocked_dense_logits_refuse_a_scratch_smaller_than_a_block() {
    use eidola_engine::spec::Bucket;
    use eidola_engine_cuda::qwen::QwenCudaExecutor;
    use qwen_common::pair::*;
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let model = fixture();
    let mut cfg = qwen_common::with_block_size(16, || cuda_config(&Geometry::plain(0)));
    cfg.buckets = vec![Bucket {
        max_seqs: 1,
        max_tokens: 1,
    }];
    let mut ex =
        QwenCudaExecutor::new(qwen_common::block_deltanet(lib), &model, cfg).expect("executor");
    let err = ex
        .dense_logits(0, &TestRng(8).tokens(20, VOCAB))
        .expect_err("refused");
    assert!(err.to_string().contains("whole block"), "{err}");
    assert!(ex.slot_state_is_zero(0));
}

/// In blocks a step prepares only the positions it computes: a decode row its host token
/// and drafts (its block's earlier positions are in the tail cache), a prefill its
/// chunk. On the eager and the rung path, at blocks of 16 and widths 0 and 3, every
/// DeltaNet layer's preparation covers exactly its runs' new positions.
#[test]
fn blocked_deltanet_prepares_only_new_positions() {
    use eidola_engine_cuda::qwen::abi::{DnChunkParams, DnSeq};
    use eidola_engine_cuda::qwen::host::HostDevice;
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath};
    use std::sync::{Arc, Mutex};
    let Some(lib) = qwen_common::library() else {
        return;
    };
    // (positions prepared, the runs' new positions) a layer.
    let seen = Arc::new(Mutex::new(Vec::<(u32, u32)>::new()));
    let make = || {
        let inner = qwen_common::host_launch(lib.clone());
        let seen = seen.clone();
        let prepared = Mutex::new(None::<u32>);
        HostDevice::block_deltanet(Box::new(move |family, params, gx, gy| {
            if family == "dn_prep" {
                *prepared.lock().expect("lock") = Some(gy);
            }
            if family == "dn_intra" {
                // SAFETY: the host device hands `dn_intra` its `DnChunkParams`, whose runs
                // are a host array of `seq_count` descriptors.
                let new: u32 = unsafe {
                    let p = &*params.cast::<DnChunkParams>();
                    std::slice::from_raw_parts(p.seqs as *const DnSeq, p.seq_count as usize)
                        .iter()
                        .map(|s| s.count - s.out_from)
                        .sum()
                };
                let got = prepared.lock().expect("lock").take().unwrap_or(u32::MAX);
                seen.lock().expect("lock").push((got, new));
            }
            inner(family, params, gx, gy);
        }))
    };
    qwen_common::with_block_size(16, || {
        for path in [DecodePath::Eager, DecodePath::Direct] {
            for k in [0u32, 3] {
                seen.lock().expect("lock").clear();
                let before = qwen_common::rung_steps();
                qwen_common::with_path(CudaGraphs::Off, path, || {
                    qwen_common::invariance::workload(&make, k)
                });
                let seen = seen.lock().expect("lock");
                assert!(
                    !seen.is_empty(),
                    "{path:?} at width {k}: no blocked layer ran"
                );
                assert!(
                    path == DecodePath::Eager || qwen_common::rung_steps() > before,
                    "{path:?} at width {k}: no step ran a rung"
                );
                let off: Vec<_> = seen.iter().filter(|(got, new)| got != new).collect();
                assert!(
                    off.is_empty(),
                    "{path:?} at width {k}: {} of {} layers prepared other than their new positions, e.g. {:?}",
                    off.len(),
                    seen.len(),
                    &off[..off.len().min(4)]
                );
            }
        }
    });
}

/// The padded rung path (a decode rung's launch sequence over its table, as a graph
/// records it, run directly: the host captures no graphs) against the CPU reference
/// executor: the same scenarios, and rungs must have run.
#[test]
fn the_padded_rung_path_matches_the_reference() {
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath};
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let make = || qwen_common::host_device(lib.clone());
    let before = qwen_common::rung_steps();
    qwen_common::with_path(CudaGraphs::Off, DecodePath::Direct, || {
        scenarios::one_request(make());
        scenarios::chunkings(&make);
        scenarios::hits_and_pressure(&make);
        scenarios::alone_and_crowded(&make);
        scenarios::empty_tails(&make);
        scenarios::draft_patterns(&make);
        scenarios::endings_inside_drafts(&make);
    });
    let n = qwen_common::rung_steps() - before;
    eprintln!("{n} steps ran a rung");
    assert!(n > 100, "only {n} steps ran a rung");
}

/// Reproducibility between calls across the paths: the same workloads eagerly and on the
/// padded rung path give the same tokens and logits bits, and every invariance scenario
/// holds on the rung path.
#[test]
fn the_rung_path_and_the_eager_path_agree() {
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath};
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let make = || qwen_common::host_device(lib.clone());
    let mut failed = Vec::new();
    let paths = [
        (CudaGraphs::Off, DecodePath::Eager),
        (CudaGraphs::Off, DecodePath::Direct),
    ];
    for (name, r) in qwen_common::invariance::across_paths(&make, &paths) {
        match r {
            Ok(what) => eprintln!("ok      {name}: {what}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    qwen_common::with_path(CudaGraphs::Off, DecodePath::Direct, || {
        for (name, r) in qwen_common::invariance::all(&make) {
            match r {
                Ok(what) => eprintln!("ok      direct {name}: {what}"),
                Err(e) => {
                    eprintln!("FAILED  direct {name}: {e}");
                    failed.push(name.to_string());
                }
            }
        }
    });
    assert!(failed.is_empty(), "{failed:?}");
}

/// The fast profile's blocked DeltaNet through the real scheduler on the host: every
/// launch but the DeltaNet's the exact profile's bodies, the DeltaNet in blocks (the intra
/// kernel's body, the state kernel's host form). Reproducibility between calls
/// (`invariance::all`, at blocks of 4 positions from KV blocks of 4) and across block
/// boundaries at blocks of 16 (`invariance::blocks`), and the same workloads on the eager
/// and the rung path: tokens and every logit's bits.
#[test]
fn blocked_deltanet_is_reproducible_through_the_scheduler() {
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath};
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let make = || qwen_common::block_deltanet(lib.clone());
    let paths = [
        (CudaGraphs::Off, DecodePath::Eager),
        (CudaGraphs::Off, DecodePath::Direct),
    ];
    let mut failed = Vec::new();
    let mut report = |name: String, r: Result<String, String>| match r {
        Ok(what) => eprintln!("ok      {name}: {what}"),
        Err(e) => {
            eprintln!("FAILED  {name}: {e}");
            failed.push(name);
        }
    };
    for (name, r) in qwen_common::invariance::all(&make) {
        report(format!("blocks of 4: {name}"), r);
    }
    for (name, r) in qwen_common::invariance::across_paths(&make, &paths) {
        report(format!("blocks of 4: {name}"), r);
    }
    let exact = || qwen_common::host_device(lib.clone());
    let deviation = qwen_common::invariance::deviation(
        &qwen_common::invariance::reference_book(&make),
        &qwen_common::invariance::reference_book(&exact),
    );
    eprintln!("the blocked DeltaNet's deviation from the exact profile: {deviation}");
    qwen_common::with_block_size(16, || {
        for (name, r) in qwen_common::invariance::blocks(&make) {
            report(format!("blocks of 16: {name}"), r);
        }
        for (name, r) in qwen_common::invariance::across_paths(&make, &paths) {
            report(format!("blocks of 16: {name}"), r);
        }
    });
    assert!(failed.is_empty(), "{failed:?}");
}

/// The fast profile's plans on every path without a device, on the fixture with 256-wide
/// heads: FA2's work lists built and packed for every step (prefill chunks, decode rows,
/// the drafter's position pages, the rungs' reservations) and each launch checked,
/// computing nothing. Its attention reads as zeros, so the logits are not the model's,
/// but they are one function of the requests on the eager and rung paths alike, which
/// must agree bit for bit.
#[test]
fn fast_profile_plans_pack_on_every_path() {
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath};
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let make = || qwen_common::planning_fast(lib.clone());
    let paths = [
        (CudaGraphs::Off, DecodePath::Eager),
        (CudaGraphs::Off, DecodePath::Direct),
    ];
    let mut failed = Vec::new();
    let results =
        qwen_common::with_fixture_256(|| qwen_common::invariance::across_paths(&make, &paths));
    for (name, r) in results {
        match r {
            Ok(what) => eprintln!("ok      {name}: {what}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// The fixture with 256-wide heads: the model crate's forward equals its goldens (the
/// released model code, pinned) bit for bit, and the exact profile on it equals the CPU
/// reference executor through the scheduler, eagerly and on the padded rung path.
#[test]
fn the_256_wide_fixture_runs_exact() {
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath};
    use eidola_engine_model::WeightSet;
    use eidola_engine_model::qwen::{LogitRows, QwenForwardOptions};
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let dir = qwen_common::pair::fixture_dir("qwen_tiny256");
    let golden = WeightSet::open_files(&[dir.join("golden.safetensors")]).expect("goldens");
    let tokens: Vec<u32> = golden
        .get("tokens")
        .and_then(|t| t.to_i64("tokens"))
        .expect("tokens")
        .into_iter()
        .map(|t| u32::try_from(t).expect("a token"))
        .collect();
    let want = golden
        .get("prefill.logits")
        .and_then(|t| t.to_f32("prefill.logits"))
        .expect("logits");
    qwen_common::with_fixture_256(|| {
        let model = qwen_common::pair::fixture();
        assert_eq!(model.config().attention.head_dim, 256);
        let opts = QwenForwardOptions {
            logits: LogitRows::All,
            keep_streams: false,
        };
        let got = model
            .forward(&mut model.new_state(), &tokens, &opts)
            .expect("forward");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&got.logits.data),
            bits(&want),
            "the forward against its goldens"
        );
        let make = || qwen_common::host_device(lib.clone());
        for (graphs, path) in [
            (CudaGraphs::Off, DecodePath::Eager),
            (CudaGraphs::Off, DecodePath::Direct),
        ] {
            qwen_common::with_path(graphs, path, || {
                scenarios::chunkings(&make);
                scenarios::alone_and_crowded(&make);
            });
        }
    });
}

/// A drafting step runs the target once: every uniform decode step at draft width `k`
/// launches the FP8 linears of one target forward (an undrafted decode step's count) and
/// the drafter's, its layer's seven (the attention's four projections and the MLP's
/// three) at each of `k` depths and its K and V projections for the canonical rewrite;
/// eagerly and on the padded rung path. Two target forwards would launch the target's
/// linears twice.
#[test]
fn a_drafting_step_runs_the_target_once() {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use eidola_engine::engine::{CacheScope, Engine};
    use eidola_engine::executor::{Executor, ExecutorError, StepInput, StepOutput};
    use eidola_engine::sampling::SamplingParams;
    use eidola_engine::spec::ModelSpec;
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath, QwenCudaExecutor};
    use qwen_common::pair::*;

    type Counts = Arc<Mutex<HashMap<&'static str, u64>>>;
    /// The FP8 linear launches of every uniform decode step, by its draft width.
    struct PerStep<E> {
        inner: E,
        counts: Counts,
        decode: Vec<(u32, u64)>,
    }
    impl<E: Executor> Executor for PerStep<E> {
        fn spec(&self) -> &ModelSpec {
            self.inner.spec()
        }
        fn execute(&mut self, step: &StepInput) -> Result<StepOutput, ExecutorError> {
            let lin8 = |c: &Counts| c.lock().expect("counts").get("lin8").copied().unwrap_or(0);
            let before = lin8(&self.counts);
            let out = self.inner.execute(step)?;
            let w = step.seqs.first().map_or(0, |e| e.num_drafts);
            let uniform = !step.seqs.is_empty()
                && step
                    .seqs
                    .iter()
                    .all(|e| e.num_tokens == 1 && e.sample && e.num_drafts == w);
            if uniform {
                self.decode.push((w, lin8(&self.counts) - before));
            }
            Ok(out)
        }
    }

    let Some(lib) = qwen_common::library() else {
        return;
    };
    let run = |k: u32, path: DecodePath| -> Vec<(u32, u64)> {
        let counts = Counts::default();
        let dev = qwen_common::counting_device(lib.clone(), counts.clone());
        let g = Geometry {
            graphs: CudaGraphs::Off,
            ..Geometry::plain(k)
        };
        let mut ex = QwenCudaExecutor::new(dev, &fixture(), cuda_config(&g)).expect("executor");
        ex.set_decode_path(path).expect("path");
        let rec = PerStep {
            inner: ex,
            counts,
            decode: Vec::new(),
        };
        let mut eng = Engine::new(rec, sched(256, k > 0)).expect("config");
        for id in 0..3u64 {
            let prompt = TestRng(40 + id).tokens(9, VOCAB);
            eng.submit(request(
                id,
                prompt,
                SamplingParams::greedy(),
                12,
                CacheScope::Private,
            ))
            .expect("submit");
        }
        let mut now = 0;
        while eng.unfinished() > 0 {
            eng.step(now).expect("step");
            now += 1;
        }
        let stats = eng.executor().inner.decode_stats();
        assert!(
            path == DecodePath::Eager || stats.direct > 0,
            "{path:?} at width {k}: no step ran a rung"
        );
        eng.executor().decode.clone()
    };
    for path in [DecodePath::Eager, DecodePath::Direct] {
        let plain = run(0, path);
        let target = plain.first().expect("decode steps").1;
        assert!(
            plain.iter().all(|&(_, n)| n == target),
            "{path:?}: undrafted decode steps launch {plain:?}"
        );
        for k in [1u32, 3] {
            let steps = run(k, path);
            let drafted: Vec<u64> = steps.iter().filter(|s| s.0 == k).map(|s| s.1).collect();
            assert!(!drafted.is_empty(), "{path:?}: no decode step drafted {k}");
            let want = target + 7 * u64::from(k) + 2;
            assert!(
                drafted.iter().all(|&n| n == want),
                "{path:?} width {k}: FP8 linear launches {drafted:?}, one target forward \
                 and the drafter's being {want} (the target alone {target})"
            );
        }
    }
}
